"""Deterministic experimental-only DSP baseline. No fitted preprocessing or ML.

Run: python dsp_baseline.py
Full windows only ('valid' convolution); no spatial padding. Threshold rule >=.
Tie policy: highest threshold within W, then smallest W across equal validation BA.
All selections are frozen and saved before any test metrics are calculated.
"""
from pathlib import Path
import hashlib
import json
import lzma
import platform
import numpy as np
import pandas as pd
import scipy
from scipy.signal import hilbert
from scipy.stats import rankdata
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results' / 'dsp_baseline'
WINDOWS = (1, 3, 5, 7)
BATCHES = [f'batch_{i:03d}' for i in range(13, 20)]
SHAPE = (480, 7168)
IMAGE_BYTES = 480 * 7168 * 2


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def process_image(image):
    """Only input is image pixels. No labels, depth, location, or source metadata."""
    require(image.shape == SHAPE, 'Unexpected image shape')
    roi = image[:, 1100:3100].astype(np.float64)
    centered = roi - roi.mean(axis=1, keepdims=True)
    envelope = np.abs(hilbert(centered, axis=1))
    line = envelope.max(axis=1)
    return roi, envelope, line


def smooth_line(line, window):
    return np.convolve(line, np.ones(window, dtype=np.float64) / window, mode='valid')


def image_scores(image):
    line = process_image(image)[2]
    return {f'score_w{w}': float(smooth_line(line, w).max()) for w in WINDOWS}


def select_parameters(validation):
    require(validation.partition.eq('validation').all(), 'Tuning accepts validation rows ONLY')
    require(validation.sample_id.is_unique, 'Duplicate tuning sample')
    labels = validation.has_flaw.to_numpy(dtype=int)
    require(set(labels) == {0, 1}, 'Both validation classes required')
    p, n = int(labels.sum()), int((labels == 0).sum())
    candidates = []
    for w in WINDOWS:
        scores = validation[f'score_w{w}'].to_numpy()
        require(np.isfinite(scores).all(), 'Nonfinite score')
        order = np.argsort(scores, kind='stable')
        sorted_scores, sorted_labels = scores[order], labels[order]
        thresholds = np.r_[np.unique(scores), np.nextafter(scores.max(), np.inf)]
        below = np.searchsorted(sorted_scores, thresholds, side='left')
        cumulative = np.r_[0, np.cumsum(sorted_labels)]
        tp = p - cumulative[below]
        tn = below - cumulative[below]
        objective = tp * n + tn * p  # exact integer ordering of balanced accuracy
        best = np.flatnonzero(objective == objective.max())[-1]
        candidates.append(dict(window=w, threshold=float(thresholds[best]),
            validation_balanced_accuracy=float(objective[best] / (2 * p * n)),
            objective=int(objective[best])))
    chosen = max(candidates, key=lambda row: (row['objective'], -row['window']))
    return chosen, candidates


def metrics(labels, scores, predictions):
    y, predicted = np.asarray(labels), np.asarray(predictions)
    tp = int(((y == 1) & (predicted == 1)).sum())
    tn = int(((y == 0) & (predicted == 0)).sum())
    fp = int(((y == 0) & (predicted == 1)).sum())
    fn = int(((y == 1) & (predicted == 0)).sum())
    p, n = tp + fn, tn + fp
    recall, specificity = tp / p, tn / n
    # Mann-Whitney rank statistic: equivalent ROC-AUC, including score ties.
    ranks = rankdata(scores, method='average')
    auc = (ranks[y == 1].sum() - p * (p + 1) / 2) / (p * n)
    return dict(TP=tp, TN=tn, FP=fp, FN=fn, accuracy=(tp + tn) / len(y),
        recall=recall, sensitivity=recall, specificity=specificity,
        precision=tp / (tp + fp) if tp + fp else 0.,
        F1=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.,
        false_positive_rate=fp / n, balanced_accuracy=(recall + specificity) / 2,
        ROC_AUC=float(auc))


def tune_scheme(manifest, scores):
    validation = manifest.loc[manifest.partition.eq('validation'),
                              ['sample_id', 'has_flaw', 'partition']].merge(
                                  scores, on='sample_id', validate='one_to_one')
    test_ids = set(manifest.loc[manifest.partition.eq('test'), 'sample_id'])
    require(not set(validation.sample_id) & test_ids, 'Test IDs entered tuning')
    selection, candidates = select_parameters(validation)
    return selection, candidates, validation.sample_id.tolist()


def validate_manifests(ordinary, holdout):
    require(len(ordinary) == 7000 and ordinary.sample_id.is_unique, 'Ordinary coverage error')
    require(set(ordinary.batch) == set(BATCHES), 'Nonexperimental or missing batch')
    require(set(holdout.fold) == set(range(1, 7)), 'Expected six folds')
    common = ['sample_id', 'batch', 'has_flaw', 'source_label', 'flaw_size', 'metadata_row']
    base = ordinary[common].sort_values('sample_id').reset_index(drop=True)
    for fold, frame in holdout.groupby('fold'):
        require(len(frame) == 7000 and frame.sample_id.is_unique, 'Fold coverage error')
        pd.testing.assert_frame_equal(base, frame[common].sort_values('sample_id').reset_index(drop=True))
        for partition, field in [('test', 'test_source'), ('validation', 'validation_source')]:
            source = frame[field].unique()
            require(len(source) == 1, 'Ambiguous held-out source')
            require(frame.loc[frame.source_label.eq(source[0]), 'partition'].eq(partition).all(), 'Source leakage')
    for frame in [ordinary, holdout]:
        require(set(frame.partition) == {'train', 'validation', 'test'}, 'Invalid partition')


def compute_scores(ordinary, dataset):
    records, examples = [], {}
    # Examples are selected from ordinary TRAINING only, before looking at scores.
    chosen = {label: ordinary.loc[ordinary.partition.eq('train') & ordinary.has_flaw.eq(label)]
              .sort_values(['batch', 'metadata_row']).iloc[0].sample_id for label in (0, 1)}
    for batch in BATCHES:
        rows = ordinary.loc[ordinary.batch.eq(batch)].sort_values('metadata_row')
        require(rows.metadata_row.tolist() == list(range(1, 1001)), 'Metadata order error')
        with lzma.open(dataset / f'{batch}.xz', 'rb') as stream:
            for row in rows.itertuples():
                payload = stream.read(IMAGE_BYTES)
                require(len(payload) == IMAGE_BYTES, 'Truncated image')
                image = np.frombuffer(payload, dtype='<i2').reshape(SHAPE)
                records.append(dict(sample_id=row.sample_id, **image_scores(image)))
                for label, sample_id in chosen.items():
                    if row.sample_id == sample_id:
                        examples[label] = (sample_id, image.copy())
                if row.metadata_row % 250 == 0:
                    print(f'{batch}: scored {row.metadata_row}/1000', flush=True)
            require(stream.read(1) == b'', 'Unexpected extra image bytes')
    return pd.DataFrame(records), examples


def distribution_plot(predictions, title, path):
    fig, ax = plt.subplots(figsize=(7, 4), layout='constrained')
    bins = np.linspace(predictions.dsp_score.min(), predictions.dsp_score.max(), 45)
    for label, name in [(0, 'No flaw'), (1, 'Flaw')]:
        ax.hist(predictions.loc[predictions.true_label.eq(label), 'dsp_score'], bins=bins,
                alpha=.55, label=name, density=True)
    ax.axvline(predictions.threshold.iloc[0], color='black', linestyle='--', label='Validation-selected threshold')
    ax.set(title=title, xlabel='DSP score (amplitude units)', ylabel='Density')
    ax.legend(fontsize=8)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def processing_plot(sample_id, image, label, window):
    roi, envelope, line = process_image(image)
    smoothed = smooth_line(line, window)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout='constrained')
    for ax, values, title, cmap in [
        (axes[0, 0], roi, 'Raw RF crop: columns 1100:3100', 'RdBu_r'),
        (axes[0, 1], envelope, 'Hilbert envelope after per-A-scan DC removal', 'magma')]:
        kwargs = {} if cmap == 'magma' else dict(vmin=-np.abs(roi).max(), vmax=np.abs(roi).max())
        im = ax.imshow(values.T, aspect='auto', extent=[0, 479, 3099, 1100], cmap=cmap, **kwargs)
        ax.set(title=title, xlabel='Scan position index', ylabel='Time/sample index')
        fig.colorbar(im, ax=ax, label='Amplitude')
    axes[1, 0].plot(line)
    axes[1, 0].set(title='Line-level maximum envelope', xlabel='Scan position index', ylabel='Amplitude')
    centers = np.arange(len(smoothed)) + (window - 1) / 2
    axes[1, 1].plot(centers, smoothed)
    axes[1, 1].axhline(smoothed.max(), color='red', linestyle='--', label=f'Score = {smoothed.max():.3f}')
    axes[1, 1].set(title=f'Spatial moving average W={window} (full windows)', xlabel='Window centre scan index', ylabel='Amplitude')
    axes[1, 1].legend()
    fig.suptitle(f'{sample_id} | {"flaw" if label else "no flaw"} | ordinary training example')
    fig.savefig(OUT / f'processing_{"flaw" if label else "no_flaw"}.png', dpi=150)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    paths = [ROOT / 'data' / 'splits' / name for name in
             ['ordinary_split.csv', 'source_holdout_folds.csv']]
    hashes = {str(path.relative_to(ROOT)): digest(path) for path in paths}
    ordinary, holdout = [pd.read_csv(path) for path in paths]
    validate_manifests(ordinary, holdout)
    folders = sorted({path.parent for path in ROOT.rglob('batch_013.xz')})
    require(len(folders) == 1, 'Ambiguous dataset location')
    scores, examples = compute_scores(ordinary, folders[0])
    require(len(scores) == 7000 and scores.sample_id.is_unique, 'Score coverage error')
    scores.to_csv(OUT / 'candidate_scores.csv', index=False)
    schemes = [('ordinary', 0, ordinary)] + [('source_holdout', int(fold), frame)
               for fold, frame in holdout.groupby('fold')]
    selections, candidate_rows, tuning_rows = [], [], []
    # Phase 1: tune all schemes exclusively on their own validation partitions.
    for scheme, fold, manifest in schemes:
        selection, candidates, tuning_ids = tune_scheme(manifest, scores)
        selections.append(dict(scheme=scheme, fold=fold, **selection))
        candidate_rows.extend(dict(scheme=scheme, fold=fold, **candidate) for candidate in candidates)
        tuning_rows.extend(dict(scheme=scheme, fold=fold, sample_id=sample_id, partition='validation')
                           for sample_id in tuning_ids)
    pd.DataFrame(selections).to_csv(OUT / 'selected_parameters.csv', index=False)
    pd.DataFrame(candidate_rows).to_csv(OUT / 'validation_candidates.csv', index=False)
    pd.DataFrame(tuning_rows).to_csv(OUT / 'tuning_sample_ids.csv', index=False)
    print('All seven parameter selections frozen. Evaluating test sets once.', flush=True)
    summaries, predictions_list, source_metrics = [], [], []
    for (scheme, fold, manifest), selection in zip(schemes, selections):
        frame = manifest.merge(scores, on='sample_id', validate='one_to_one')
        frame['dsp_score'] = frame[f'score_w{selection["window"]}']
        frame['predicted_label'] = (frame.dsp_score >= selection['threshold']).astype(int)
        frame['window'], frame['threshold'] = selection['window'], selection['threshold']
        frame['scheme'], frame['fold'] = scheme, fold
        frame = frame.rename(columns={'has_flaw': 'true_label', 'source_label': 'source_flaw'})
        test = frame.loc[frame.partition.eq('test')]
        expected_test = manifest.loc[manifest.partition.eq('test'), 'sample_id']
        require(test.sample_id.is_unique and set(test.sample_id) == set(expected_test), 'Missing/duplicate test prediction')
        held = manifest.test_source.iloc[0] if scheme == 'source_holdout' else 'mixed'
        sizes = sorted(test.loc[test.true_label.eq(1), 'flaw_size'].unique())
        summaries.append(dict(**selection, test_source=held, flaw_size=';'.join(map(str, sizes)),
            n_test=len(test), **metrics(test.true_label, test.dsp_score, test.predicted_label)))
        for source, group in test.loc[test.true_label.eq(1)].groupby('source_flaw'):
            source_metrics.append(dict(scheme=scheme, fold=fold, source_flaw=source,
                flaw_size=float(group.flaw_size.iloc[0]), n_flaw=len(group),
                TP=int(group.predicted_label.sum()), FN=int((group.predicted_label == 0).sum()),
                recall=float(group.predicted_label.mean())))
        columns = ['sample_id', 'batch', 'true_label', 'source_flaw', 'flaw_size', 'dsp_score',
                   'predicted_label', 'partition', 'scheme', 'fold', 'window', 'threshold']
        predictions_list.append(frame[columns])
        prefix = f'{scheme}_{fold}'
        for suffix, mask in [('train_validation', frame.partition.isin(['train', 'validation'])),
                             ('test', frame.partition.eq('test'))]:
            distribution_plot(frame.loc[mask], f'{scheme} fold {fold}: {suffix}', OUT / f'{prefix}_{suffix}_scores.png')
    summary = pd.DataFrame(summaries)
    predictions = pd.concat(predictions_list, ignore_index=True)
    summary.to_csv(OUT / 'dsp_summary.csv', index=False)
    predictions.to_csv(OUT / 'per_sample_results.csv', index=False)
    pd.DataFrame(source_metrics).to_csv(OUT / 'per_source_recall.csv', index=False)
    errors = predictions.loc[predictions.partition.eq('test') & predictions.true_label.ne(predictions.predicted_label)].copy()
    errors['error_type'] = np.where(errors.true_label.eq(1), 'FN', 'FP')
    errors.to_csv(OUT / 'test_failure_cases.csv', index=False)
    for label, (sample_id, image) in examples.items():
        processing_plot(sample_id, image, label, selections[0]['window'])
    row = summary.iloc[0]
    matrix = np.array([[row.TN, row.FP], [row.FN, row.TP]], dtype=int)
    fig, ax = plt.subplots(figsize=(5, 4), layout='constrained')
    ax.imshow(matrix, cmap='Blues')
    for (i, j), value in np.ndenumerate(matrix):
        ax.text(j, i, str(value), ha='center', va='center', color='white' if value > matrix.max()/2 else 'black')
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=['No flaw', 'Flaw'], yticklabels=['No flaw', 'Flaw'],
           xlabel='Predicted', ylabel='True', title='Ordinary split: test confusion matrix')
    fig.savefig(OUT / 'ordinary_confusion_matrix.png', dpi=150)
    plt.close(fig)
    source = pd.DataFrame(source_metrics).query("scheme == 'source_holdout'")
    fig, ax = plt.subplots(figsize=(10, 4), layout='constrained')
    bars = ax.bar(source.source_flaw, source.recall)
    ax.bar_label(bars, labels=[f'{value:.3f}' for value in source.recall], padding=3)
    ax.set(ylim=(0, 1.1), ylabel='Test recall', title='Unseen-source DSP recall (validation-selected W and threshold)')
    fig.savefig(OUT / 'source_holdout_recall.png', dpi=150)
    plt.close(fig)
    require(hashes == {str(path.relative_to(ROOT)): digest(path) for path in paths}, 'Split files modified')
    (OUT / 'checks.json').write_text(json.dumps(dict(split_sha256=hashes,
        split_files_unchanged=True, validation_only_tuning=True, all_test_samples_predicted_once=True,
        detector_input='image array only; no depth/location metadata loaded',
        preprocessing='per-image per-A-scan DC mean; no learned/global statistics',
        roi=[1100, 3100], windows=list(WINDOWS), spatial_boundary='valid: full windows only',
        threshold_rule='score >= threshold', tie_policy='highest threshold, then smallest W',
        python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__), indent=2))
    print(summary.to_string(index=False))


if __name__ == '__main__':
    main()
