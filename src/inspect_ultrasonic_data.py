"""Inspect experimental XZ streams, selected images and metadata. No DSP/ML.

Decoding hypothesis supported by byte counts and waveform continuity:
headerless little-endian int16, C-order (image, scan index, sound-path sample).
The README does not explicitly specify dtype, endianness, or serialization order.
Full streams are checked to EOF; statistics cover only selected images.
"""
from pathlib import Path
import json
import lzma
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
SHAPE = (480, 7168)
DTYPE = np.dtype('<i2')
IMAGE_BYTES = int(np.prod(SHAPE)) * DTYPE.itemsize
SOURCES = ['no_flaw', 'P41_01', 'P41_02', 'P41_03', 'P41_04', 'P41_05', 'P41_06_notch']
OUT = ROOT / 'results' / 'data_inspection'


def stats(image):
    return dict(minimum=int(image.min()), maximum=int(image.max()),
                mean=float(image.mean()), std=float(image.std()),
                negative_fraction=float((image < 0).mean()),
                nan_count=int(np.isnan(image).sum()), inf_count=int(np.isinf(image).sum()))


def check_representative_alignment(images, representatives):
    """Compare stored images, without filtering or implementing a detector."""
    evidence = []
    reference = images['no_flaw']
    for label in SOURCES[1:]:
        scan, time = np.nonzero(images[label] != reference)
        row = representatives[label]
        evidence.append(dict(sample_id=row.sample_id, category=label,
            reference_sample_id=representatives['no_flaw'].sample_id,
            metadata_location=float(row.location), metadata_depth=float(row.depth),
            difference_scan_start=int(scan.min()), difference_scan_end=int(scan.max()),
            difference_time_start=int(time.min()), difference_time_end=int(time.max()),
            origin_matches_metadata=bool(scan.min() == int(row.location)
                                         and time.min() == int(row.depth))))
    result = pd.DataFrame(evidence)
    result.to_csv(OUT / 'representative_alignment.csv', index=False)
    if not result.origin_matches_metadata.all():
        raise ValueError('Representative alignment does not match the proposed decoding')
    print('All six representative difference-region origins match metadata location/depth.')


def plot_example(image, row, limit):
    fig, axes = plt.subplots(1, 3, figsize=(15, 6), layout='constrained',
                             gridspec_kw={'width_ratios': [1, 1, 1.25]})
    for ax, part, extent, title in [
        (axes[0], image, [-.5, 479.5, 7167.5, -.5], 'Full image (480 x 7168)'),
        (axes[1], image[:, 1100:3100], [-.5, 479.5, 3099.5, 1099.5],
         'README region: [:, 1100:3100]'),
    ]:
        handle = ax.imshow(part.T, aspect='auto', origin='upper', extent=extent,
                           cmap='RdBu_r', vmin=-limit, vmax=limit, interpolation='nearest')
        ax.set(title=title, xlabel='Scan index (axis 0)', ylabel='Sound-path sample index (axis 1)')
    for boundary in [1100, 3100]:
        axes[0].axhline(boundary, color='limegreen', linewidth=1)
    scan = int(np.clip(round(row.location), 0, 479)) if row.has_flaw else 240
    axes[2].plot(np.arange(1100, 3100), image[scan, 1100:3100], linewidth=.6)
    axes[2].axhline(0, color='grey', linewidth=.5)
    axes[2].set(title=f'Unprocessed signed trace: scan {scan}',
                xlabel='Sound-path sample index', ylabel='Stored amplitude (uncalibrated)')
    if row.has_flaw:
        axes[1].plot(row.location, row.depth, marker='+', color='limegreen', markersize=10)
    fig.colorbar(handle, ax=axes[:2], shrink=.7, label='Stored amplitude; common display limits')
    fig.suptitle(f'{row.sample_id} | {row.category} | original size={row.flaw_size}\n'
                 'Axis/row order inferred; green + is metadata location/depth (when flawed)')
    fig.savefig(OUT / f'{row.category}.png', dpi=150)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    metadata = pd.read_csv(ROOT / 'data' / 'metadata.csv')
    folders = sorted({p.parent for p in ROOT.rglob('batch_013.xz')})
    if len(folders) != 1:
        raise ValueError(f'Expected one dataset folder, found {folders}')
    records, images, representatives = [], {}, {}
    format_checks = []
    for number in range(13, 20):
        batch = f'batch_{number:03d}'
        rows = metadata.loc[metadata.batch.eq(batch)].sort_values('metadata_row').copy()
        if len(rows) != 1000 or rows.metadata_row.tolist() != list(range(1, 1001)):
            raise ValueError(f'{batch}: metadata row count/order mismatch')
        rows['category'] = rows.flaw_type.where(rows.has_flaw.eq(1), 'no_flaw')
        rows = rows.reset_index(drop=True)
        # One of each label for comparable composition; four spaced rows for coverage.
        first = {label: int(rows.index[rows.category.eq(label)][0]) for label in SOURCES}
        selected = set(first.values()) | {249, 499, 749, 999}
        image_count = 0
        with lzma.open(folders[0] / f'{batch}.xz', 'rb') as stream:
            while True:
                payload = stream.read(IMAGE_BYTES)
                if not payload:
                    break
                if len(payload) != IMAGE_BYTES:
                    raise ValueError(f'{batch}: partial image payload at {image_count}')
                if image_count in selected:
                    image = np.frombuffer(payload, dtype=DTYPE).reshape(SHAPE)
                    row = rows.iloc[image_count]
                    record = dict(batch=batch, sample_id=row.sample_id,
                        metadata_row=int(row.metadata_row), image_index=image_count,
                        category=row.category, matched_category_sample=image_count in first.values(),
                        shape='480x7168', dtype='int16', byte_order='little',
                        decode_status='data-supported; serialization not explicitly documented',
                        **stats(image))
                    record['crop_std'] = float(image[:, 1100:3100].std())
                    records.append(record)
                    if number == 13 and image_count in first.values():
                        images[row.category] = image.copy()
                        representatives[row.category] = row
                    if image_count == 0:
                        for dtype in ['<i2', '>i2', '<u2']:
                            alternative = np.frombuffer(payload, dtype=dtype).reshape(SHAPE).astype(float)
                            format_checks.append(dict(batch=batch, interpretation=dtype,
                                adjacent_sample_mean_abs_difference=float(np.abs(np.diff(alternative, axis=1)).mean())))
                image_count += 1
        if image_count != len(rows):
            raise ValueError(f'{batch}: {image_count} decoded images vs {len(rows)} metadata rows')
        for record in records:
            if record['batch'] == batch:
                record.update(images_per_batch=image_count, decompressed_bytes=image_count * IMAGE_BYTES,
                              stream_verified_to_eof=True)
        print(f'{batch}: {image_count} image-sized records; {len(selected)} inspected; EOF verified', flush=True)

    audit = pd.DataFrame(records)
    audit.to_csv(ROOT / 'data' / 'image_data_audit.csv', index=False)
    pd.DataFrame(format_checks).to_csv(OUT / 'decode_comparison.csv', index=False)
    # Shared limits across representative panels; only display clipping, no signal processing.
    limit = max(float(np.quantile(np.abs(im.astype(float)), .995)) for im in images.values())
    for label in SOURCES:
        plot_example(images[label], representatives[label], limit)
    check_representative_alignment(images, representatives)
    matched = audit.loc[audit.matched_category_sample]
    summary = matched.groupby('batch').agg(
        n_inspected=('sample_id', 'size'), minimum=('minimum', 'min'), maximum=('maximum', 'max'),
        mean=('mean', 'mean'), mean_image_std=('std', 'mean'), mean_crop_std=('crop_std', 'mean'))
    summary.to_csv(OUT / 'batch_scaling.csv')
    ratio = float(summary.mean_image_std.max() / summary.mean_image_std.min())
    crop_ratio = float(summary.mean_crop_std.max() / summary.mean_crop_std.min())
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout='constrained')
    for category in SOURCES:
        subset = matched.loc[matched.category.eq(category)]
        axes[0].plot(subset.batch.str[-3:], subset['std'], marker='.', label=category)
        axes[1].plot(subset.batch.str[-3:], subset.crop_std, marker='.')
    axes[0].set(title='Full image standard deviation', xlabel='Experimental batch', ylabel='Stored amplitude units')
    axes[1].set(title='README crop standard deviation', xlabel='Experimental batch', ylabel='Stored amplitude units')
    axes[0].legend(fontsize=7)
    fig.savefig(OUT / 'batch_scaling.png', dpi=150)
    plt.close(fig)
    print('\nMatched seven-category scaling comparison:\n' + summary.to_string())
    print(f'Largest/smallest mean image std: {ratio:.5f}; crop std: {crop_ratio:.5f}')
    print(f'Inspected {len(audit)} images; sampled range {audit.minimum.min()} to {audit.maximum.max()}')
    print(f'Common plot limits: +/-{limit:g}; values beyond these limits are visually clipped only.')
    (OUT / 'run_summary.json').write_text(json.dumps(dict(
        selected_images=len(audit), images_per_batch=1000, image_shape=list(SHAPE),
        dtype='<i2', common_plot_limit=limit, mean_std_ratio=ratio,
        mean_crop_std_ratio=crop_ratio, sampled_min=int(audit.minimum.min()),
        sampled_max=int(audit.maximum.max())), indent=2))


if __name__ == '__main__':
    main()
