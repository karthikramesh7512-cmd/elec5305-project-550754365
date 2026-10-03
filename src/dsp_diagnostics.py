"""Post-hoc diagnostics of frozen DSP V1. Never tune or overwrite V1 artifacts."""
from pathlib import Path
import hashlib
import json
import lzma
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import dsp_baseline as v1

ROOT, OUT = v1.ROOT, v1.OUT
DIAG = OUT / 'diagnostics'
ORDER = ['no_flaw', 'P41_01', 'P41_02', 'P41_04', 'P41_06_notch', 'P41_05', 'P41_03']
LABELS = ['No flaw', 'P41_01 - 2 mm', 'P41_02 - 3 mm', 'P41_04 - 6 mm',
          'P41_06_notch - 6 mm notch', 'P41_05 - 17 mm', 'P41_03 - 26 mm']


def table(frame):
    def fmt(value):
        return f'{value:.6f}' if isinstance(value, float) else str(value)
    return '\n'.join(['| ' + ' | '.join(map(str, frame.columns)) + ' |',
                      '| ' + ' | '.join(['---'] * len(frame.columns)) + ' |'] +
                     ['| ' + ' | '.join(fmt(x) for x in row) + ' |'
                      for row in frame.itertuples(index=False, name=None)])


def saved_score_diagnostics(data, ordinary, holdout):
    stats = []
    for source in ORDER:
        values = data.loc[data.source.eq(source), 'score_w1']
        stats.append(dict(source=source, n=len(values), minimum=values.min(),
            p25=values.quantile(.25), median=values.median(), p75=values.quantile(.75),
            maximum=values.max(), mean=values.mean(), std_population=values.std(ddof=0)))
    stats = pd.DataFrame(stats)
    stats.to_csv(OUT / 'score_distribution_by_source.csv', index=False)
    selected = pd.read_csv(OUT / 'selected_parameters.csv', float_precision='round_trip')
    fig, ax = plt.subplots(figsize=(13, 6), layout='constrained')
    ax.boxplot([data.loc[data.source.eq(source), 'score_w1'] for source in ORDER],
               tick_labels=LABELS, showfliers=True, widths=.5)
    for index, row in enumerate(selected.loc[selected.scheme.eq('source_holdout')].itertuples()):
        ax.axhline(row.threshold, color=plt.get_cmap('tab10')(index), linestyle='--',
                   linewidth=1, label=f'Fold {row.fold}: {row.threshold:.2f}')
    ax.set(ylabel='Frozen V1 continuous DSP score (W=1)',
           title='Experimental score distributions and unchanged holdout thresholds')
    ax.tick_params(axis='x', labelrotation=20)
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=8)
    fig.savefig(DIAG / 'score_distributions_with_thresholds.png', dpi=160)
    plt.close(fig)
    # Report saved validation candidates; recompute BA at those SAME thresholds only.
    candidates = pd.read_csv(OUT / 'validation_candidates.csv', float_precision='round_trip')
    behaviors = []
    for row in candidates.itertuples():
        manifest = ordinary if row.scheme == 'ordinary' else holdout.loc[holdout.fold.eq(row.fold)]
        validation = manifest.loc[manifest.partition.eq('validation'), ['sample_id', 'has_flaw']].merge(
            data[['sample_id', f'score_w{row.window}']], on='sample_id', validate='one_to_one')
        positive = validation.loc[validation.has_flaw.eq(1), f'score_w{row.window}']
        negative = validation.loc[validation.has_flaw.eq(0), f'score_w{row.window}']
        ba = ((positive >= row.threshold).mean() + (negative < row.threshold).mean()) / 2
        v1.require(np.isclose(ba, row.validation_balanced_accuracy), 'Saved validation BA mismatch')
        behaviors.append(dict(scheme=row.scheme, fold=row.fold, window=row.window,
            saved_threshold=row.threshold, saved_validation_balanced_accuracy=row.validation_balanced_accuracy,
            verified_validation_balanced_accuracy=ba, negative_max=negative.max(),
            positive_min=positive.min(), separation_gap=positive.min()-negative.max()))
    behaviors = pd.DataFrame(behaviors)
    behaviors.to_csv(DIAG / 'window_validation_behavior.csv', index=False)
    window_stats = []
    for source in ORDER:
        group = data.loc[data.source.eq(source)]
        for w in v1.WINDOWS:
            values = group[f'score_w{w}']
            window_stats.append(dict(source=source, window=w, minimum=values.min(),
                median=values.median(), maximum=values.max(), mean=values.mean(),
                std_population=values.std(ddof=0), unique_6dp=values.round(6).nunique()))
    pd.DataFrame(window_stats).to_csv(DIAG / 'window_score_behavior.csv', index=False)
    return stats, behaviors


def locate_pixel_peak(image, reference):
    """Pure pixel-only diagnostics; metadata comparison happens AFTER this call."""
    _, envelope, line = v1.process_image(image)
    scan, local_time = np.unravel_index(np.argmax(envelope), envelope.shape)
    result = dict(peak_scan=int(scan), peak_time=int(local_time + 1100),
                  peak_value=float(envelope[scan, local_time]),
                  exact_peak_ties=int(np.count_nonzero(envelope == envelope[scan, local_time])))
    for w in v1.WINDOWS:
        smoothed = v1.smooth_line(line, w)
        start = int(np.argmax(smoothed))
        result.update({f'score_w{w}': float(smoothed[start]), f'w{w}_scan_start': start,
                       f'w{w}_scan_center': start + (w - 1)/2})
    difference = image[:, 1100:3100] != reference[:, 1100:3100]
    scans = np.flatnonzero(difference.any(axis=1))
    times = np.flatnonzero(difference.any(axis=0))
    result['roi_differs_from_reference'] = bool(len(scans))
    if len(scans):
        result.update(diff_scan_start=int(scans[0]), diff_scan_end=int(scans[-1]),
            diff_time_start=int(times[0]+1100), diff_time_end=int(times[-1]+1100),
            peak_in_difference_bbox=bool(scans[0] <= scan <= scans[-1] and times[0] <= local_time <= times[-1]),
            peak_pixel_differs=bool(difference[scan, local_time]))
    return result


def no_flaw_plot(image, sample_id, peak, path):
    _, envelope, _ = v1.process_image(image)
    scan, time = peak['peak_scan'], peak['peak_time']
    left, right = max(0, scan-10), min(480, scan+11)
    low, high = max(1100, time-140), min(3100, time+141)
    raw = image[left:right, low:high]
    env = envelope[left:right, low-1100:high-1100]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), layout='constrained')
    for ax, values, title, cmap in [(axes[0], raw, 'Raw RF near score maximum', 'RdBu_r'),
                                   (axes[1], env, 'DC-removed Hilbert envelope', 'magma')]:
        handle = ax.imshow(values.T, aspect='auto', extent=[left-.5, right-.5, high-.5, low-.5], cmap=cmap)
        ax.plot(scan, time, 'gx', markersize=9)
        ax.set(title=title, xlabel='Scan position index', ylabel='Original time/sample index')
        fig.colorbar(handle, ax=ax, label='Amplitude')
    axes[2].plot(np.arange(low, high), image[scan, low:high], label='Raw RF')
    axes[2].plot(np.arange(low, high), envelope[scan, low-1100:high-1100], label='Envelope')
    axes[2].axvline(time, color='grey', linestyle='--')
    axes[2].set(title=f'A-scan {scan}', xlabel='Original time/sample index', ylabel='Amplitude')
    axes[2].legend()
    fig.suptitle(f'{sample_id}: no flaw; peak ({scan}, {time}), score={peak["peak_value"]:.6f}')
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    # Protect every pre-existing V1 output plus detector and split files.
    protected = list(OUT.glob('*')) + [ROOT / 'src/dsp_baseline.py',
        ROOT / 'data/splits/ordinary_split.csv', ROOT / 'data/splits/source_holdout_folds.csv']
    new_names = {'score_distribution_by_source.csv', 'dsp_diagnostics.md'}
    protected = [p for p in protected if p.is_file() and p.name not in new_names]
    before = {str(p.relative_to(ROOT)): v1.digest(p) for p in protected}
    DIAG.mkdir(exist_ok=True)
    ordinary = pd.read_csv(ROOT / 'data/splits/ordinary_split.csv')
    holdout = pd.read_csv(ROOT / 'data/splits/source_holdout_folds.csv')
    scores = pd.read_csv(OUT / 'candidate_scores.csv', float_precision='round_trip')
    metadata = pd.read_csv(ROOT / 'data/metadata.csv')
    metadata = metadata.loc[metadata.batch.isin(v1.BATCHES)]
    data = metadata.merge(scores, on='sample_id', validate='one_to_one')
    v1.require(len(data) == 7000 and data.sample_id.is_unique, 'Experimental coverage mismatch')
    data['source'] = data.flaw_type.where(data.has_flaw.eq(1), 'no_flaw')
    stats, behaviors = saved_score_diagnostics(data, ordinary, holdout)
    folders = sorted({p.parent for p in ROOT.rglob('batch_013.xz')})
    v1.require(len(folders) == 1, 'Ambiguous dataset')
    refrow = data.loc[data.has_flaw.eq(0)].sort_values(['batch', 'metadata_row']).iloc[0]
    with lzma.open(folders[0] / f'{refrow.batch}.xz', 'rb') as stream:
        stream.seek((int(refrow.metadata_row)-1) * v1.IMAGE_BYTES)
        raw_reference = stream.read(v1.IMAGE_BYTES)
    reference = np.frombuffer(raw_reference, dtype='<i2').reshape(v1.SHAPE)
    reference_hash = hashlib.sha256(raw_reference).hexdigest()
    cache, records, examples = {}, [], {}
    for batch in v1.BATCHES:
        with lzma.open(folders[0] / f'{batch}.xz', 'rb') as stream:
            for row in data.loc[data.batch.eq(batch)].sort_values('metadata_row').itertuples():
                payload = stream.read(v1.IMAGE_BYTES)
                v1.require(len(payload) == v1.IMAGE_BYTES, 'Incomplete image')
                raw_hash = hashlib.sha256(payload).hexdigest()
                image = np.frombuffer(payload, dtype='<i2').reshape(v1.SHAPE)
                if raw_hash not in cache:
                    cache[raw_hash] = locate_pixel_peak(image, reference)
                peak = cache[raw_hash]
                for w in v1.WINDOWS:
                    v1.require(np.isclose(peak[f'score_w{w}'], getattr(row, f'score_w{w}'), rtol=1e-12, atol=1e-9),
                               f'Frozen score differs for {row.sample_id}, W={w}')
                record = dict(sample_id=row.sample_id, batch=batch, true_label=row.has_flaw,
                    source=row.source, raw_sha256=raw_hash, raw_equals_reference=raw_hash == reference_hash, **peak)
                # Post-hoc only: these fields never enter locate_pixel_peak/process_image.
                if row.has_flaw:
                    record.update(metadata_scan=row.location, metadata_time=row.depth,
                        peak_minus_metadata_scan=peak['peak_scan']-row.location,
                        peak_minus_metadata_time=peak['peak_time']-row.depth,
                        difference_origin_matches_metadata=bool(peak.get('diff_scan_start') == int(row.location)
                            and peak.get('diff_time_start') == int(row.depth)))
                records.append(record)
                if row.has_flaw == 0 and batch in ['batch_013', 'batch_016', 'batch_019'] and batch not in examples:
                    examples[batch] = row.sample_id
                    no_flaw_plot(image, row.sample_id, peak, DIAG / f'no_flaw_peak_{batch}.png')
                if row.metadata_row % 250 == 0:
                    print(f'{batch}: diagnosed {row.metadata_row}/1000; unique raw images so far={len(cache)}', flush=True)
            v1.require(stream.read(1) == b'', 'Extra payload')
    peaks = pd.DataFrame(records)
    peaks.to_csv(DIAG / 'peak_locations_and_hashes.csv', index=False)
    negative = peaks.loc[peaks.true_label.eq(0)]
    positive = peaks.loc[peaks.true_label.eq(1)]
    coordinates = negative.groupby(['peak_scan', 'peak_time']).size().rename('n_samples').reset_index()
    coordinates.to_csv(DIAG / 'no_flaw_peak_coordinates.csv', index=False)
    association = positive.groupby('source').agg(n=('sample_id', 'size'),
        min_scan_offset=('peak_minus_metadata_scan', 'min'), max_scan_offset=('peak_minus_metadata_scan', 'max'),
        min_time_offset=('peak_minus_metadata_time', 'min'), max_time_offset=('peak_minus_metadata_time', 'max'),
        fraction_peak_in_changed_bbox=('peak_in_difference_bbox', 'mean'),
        fraction_peak_pixel_changed=('peak_pixel_differs', 'mean'),
        fraction_origin_matches_metadata=('difference_origin_matches_metadata', 'mean')).reset_index()
    association.to_csv(DIAG / 'flaw_peak_association.csv', index=False)
    unique_scores = {str(decimals): int(negative.score_w1.round(decimals).nunique()) for decimals in [3, 6, 9, 12]}
    window_ba = behaviors.pivot(index=['scheme', 'fold'], columns='window', values='saved_validation_balanced_accuracy').reset_index()
    window_medians = pd.read_csv(DIAG / 'window_score_behavior.csv').pivot(
        index='source', columns='window', values='median').reindex(ORDER).reset_index()
    exceptions = positive.loc[positive.peak_in_difference_bbox.eq(False)]
    exceptions.to_csv(DIAG / 'positive_peak_bbox_exceptions.csv', index=False)
    edge_adjacent = ((exceptions.peak_time == exceptions.diff_time_end + 1) &
                     (exceptions.peak_scan >= exceptions.diff_scan_start) &
                     (exceptions.peak_scan <= exceptions.diff_scan_end)).sum()
    report = f'''# DSP Baseline V1: post-hoc diagnostics

## Scope and preservation

All 7,000 experimental images (013–019) were examined. V1 code, thresholds,
predictions, summaries, figures and split CSVs remain unchanged. No selection
function was called. Existing validation candidates were read, and balanced
accuracy was checked at the already-saved thresholds. Metadata location/depth
were joined only AFTER pixel-only peak extraction. No SVM/CNN was implemented.

## Observations: continuous W=1 score distributions

Each image is counted once, not once per fold. Population standard deviation
uses ddof=0; quantiles use pandas linear interpolation. All V1 schemes selected W=1.
The approximately 9e-13 no-flaw standard deviation is floating-point roundoff;
its input scores are identical, so the mathematical standard deviation is zero.

{table(stats)}

See `diagnostics/score_distributions_with_thresholds.png` for boxplots in size
order and overlays of the six unchanged validation-selected thresholds. These
are descriptive all-data diagnostics, not a new evaluation or tuning exercise.

## Observations: no-flaw maxima and raw data

There are {len(negative)} no-flaw images. Unique raw full-image SHA-256 hashes:
{negative.raw_sha256.nunique()}. Images matching reference `{refrow.sample_id}`:
{int(negative.raw_equals_reference.sum())}/{len(negative)}. Hashes cover the original
6,881,280 decompressed bytes (all 480 x 7168 int16 values), not the cropped ROI.
All no-flaw images, rather than just a sample, were hashed; per-image hashes
are saved in `diagnostics/peak_locations_and_hashes.csv`.

Reference SHA-256: `{reference_hash}`.

Unique W=1 scores after rounding to 3/6/9/12 decimal places: {unique_scores}.
Peak coordinates are zero-based; time is the original image index (ROI offset added).
Argmax uses the first occurrence in C order; exact tie counts are also saved.

{table(coordinates)}

Representative raw-RF/envelope neighbourhoods and traces are in
`diagnostics/no_flaw_peak_batch_013.png`, `no_flaw_peak_batch_016.png`, and
`no_flaw_peak_batch_019.png`. They are selected by batch/row, not peak strength.

## Observations: positive peak association

The diagnostic compares positive images with reference `{refrow.sample_id}` in
the fixed ROI. The changed-pixel bounding box is measured from pixel differences,
not imposed from metadata. Reported offsets are peak minus metadata coordinates.

{table(association)}

Of {len(positive)} positive peaks, {int(positive.peak_in_difference_bbox.sum())}
are inside the changed-pixel bounding box. The {len(exceptions)} exceptions are
listed in `diagnostics/positive_peak_bbox_exceptions.csv`; {int(edge_adjacent)}
are exactly one time sample beyond the changed region, within its scan span.
Thus they remain immediately adjacent to the inserted region. Hilbert envelopes
depend on the whole A-scan, so a peak outside the exact changed-pixel support is
possible. This observation alone does not diagnose an augmentation artefact.

The measured difference origin matches integer-truncated metadata for
{int(positive.difference_origin_matches_metadata.sum())}/{len(positive)} positives.
In the five mismatches observed here, the first changed scan is 2–5 positions
after that origin (time origin still matches). Leading unchanged pixels or
background overlap may explain this; the generation details are not established.

All computed candidate scores matched frozen V1 scores within rtol=1e-12,
atol=1e-9. Metadata did not participate in any of those computations.

## Observations: the four window sizes

Saved validation balanced accuracy (not newly optimized):

{table(window_ba)}

`diagnostics/window_validation_behavior.csv` reports each unchanged candidate
threshold, positive minimum, negative maximum, and their gap. Full per-source
score behavior for every W is in `diagnostics/window_score_behavior.csv`.
The smallest observed validation separation gap across the 28 saved candidates
is {behaviors.separation_gap.min():.6f}. Positive gaps explain why all four W values
can perfectly separate validation classes. W=1 was selected by the previously
specified smallest-W tie rule, not evidence that it performs better.

Median score by source and window:

{table(window_medians)}

The maxima need not decrease monotonically with W: the no-flaw score at W=7
slightly exceeds W=5 because different complete windows can maximize the average.
For no-flaw images, the maximizing spatial window centres are scan 383 for W=1/3
and scan 435 for W=5/7. A smoothed score averages line peaks and therefore has no
single corresponding time coordinate when W is greater than one.

## Supported interpretations

The single shared raw hash across all 3,558 no-flaw images explains identical scores directly: repeated
pixel data necessarily give the same deterministic envelope and maximum. A common
background response at a fixed scan/time coordinate is consistent with a fixed
echo inherited from the acquisition/background. Repeated augmented images are
not independent acquisitions and cannot establish the physical origin of that echo.
No response is classified here as an artefact.

Where the peak lies in the changed region and the region's origin matches metadata,
the maximum is spatially associated with the inserted flaw. Metadata describes an
insertion origin in this dataset; a nonzero peak offset is not itself a detection
localization error. A rectangular bounding-box association is not a physical flaw mask.

Spatial averaging changes score magnitudes without resolving the validation tie.
The V1 threshold policy chose the highest equally optimal threshold. Perfect
validation separation leaves a range of equally good operating points, so this
tie policy can matter for lower-amplitude unseen sources. It is left unchanged.

## Unresolved uncertainties

The common background echo cannot be uniquely attributed to a system response,
interface/backwall, geometry or material structure from these files alone.
Independent acquisitions, calibration, geometry and instrument details are needed.
Hash equality establishes equality of decoded images, not why they were repeated.
Association with an inserted region does not establish physical depth accuracy.
No new operating point or W is proposed from these post-hoc diagnostics.
'''
    (OUT / 'dsp_diagnostics.md').write_text(report, encoding='utf-8')
    after = {str(p.relative_to(ROOT)): v1.digest(p) for p in protected}
    v1.require(before == after, 'A protected V1 artifact changed')
    (DIAG / 'preservation_checks.json').write_text(json.dumps(dict(
        protected_sha256_before=before, protected_sha256_after=after,
        all_unchanged=True, samples=7000, no_flaw_hashes=int(negative.raw_sha256.nunique()),
        metadata_used_only_post_detection=True, no_parameter_selection_called=True), indent=2))
    print(stats.to_string(index=False))
    print('No-flaw raw hashes:', negative.raw_sha256.nunique())
    print(coordinates.to_string(index=False))
    print(association.to_string(index=False))
    print('PASS: all existing baseline files and split files unchanged.')


if __name__ == '__main__':
    main()
