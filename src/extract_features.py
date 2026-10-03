"""Fixed pixel-only engineered features for experimental ultrasonic images."""
from pathlib import Path
import hashlib
import lzma
import numpy as np
import pandas as pd
from scipy.signal import hilbert, find_peaks

ROOT = Path(__file__).resolve().parents[1]
FEATURES = ['max_envelope', 'p95_envelope', 'mean_envelope', 'rf_rms',
    'rf_energy', 'envelope_energy', 'crest_factor', 'strong_peak_count',
    'spatial_persistence', 'strongest_response_width', 'line_peak_std', 'line_peak_mean']


def features_from_image(image):
    if image.shape != (480, 7168):
        raise ValueError('Expected (480,7168)')
    rf = image[:, 1100:3100].astype(np.float64)
    rf -= rf.mean(axis=1, keepdims=True)
    envelope = np.abs(hilbert(rf, axis=1))
    line = envelope.max(axis=1)
    maximum = float(line.max())
    energy = float(np.square(rf).sum())
    rms = float(np.sqrt(energy / rf.size))
    active = line >= .5 * maximum if maximum else np.zeros(480, dtype=bool)
    changes = np.diff(np.r_[False, active, False].astype(int))
    starts, ends = np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)
    persistence = int((ends - starts).max()) if len(starts) else 0
    strongest = int(np.argmax(line))
    containing = np.flatnonzero((starts <= strongest) & (ends > strongest))
    width = 0.
    if len(containing):
        k = containing[0]
        left, right = starts[k], ends[k] - 1
        # Linear interpolation at half maximum; boundary-clipped widths use
        # sample-cell edges (-0.5,479.5) if a crossing is outside the image.
        lcross = left - .5 if left == 0 else left-1 + (.5*maximum-line[left-1])/(line[left]-line[left-1])
        rcross = right + .5 if right == 479 else right + (line[right]-.5*maximum)/(line[right]-line[right+1])
        width = float(rcross-lcross)
    # Zero padding makes endpoints eligible local peaks; fixed definition, not tuned.
    peaks, _ = find_peaks(np.r_[0., line, 0.], height=.5*maximum,
                         prominence=.1*maximum, distance=3) if maximum else ([], {})
    values = [maximum, float(np.quantile(envelope, .95)), float(envelope.mean()), rms,
        energy, float(np.square(envelope).sum()), float(np.abs(rf).max()/rms) if rms else 0.,
        len(peaks), persistence, width, float(line.std()), float(line.mean())]
    return dict(zip(FEATURES, values))


def main():
    metadata = pd.read_csv(ROOT / 'data/splits/ordinary_split.csv')
    folders = sorted({p.parent for p in ROOT.rglob('batch_013.xz')})
    if len(folders) != 1:
        raise ValueError('Ambiguous dataset folder')
    records, cache = [], {}
    for number in range(13, 20):
        batch = f'batch_{number:03d}'
        rows = metadata.loc[metadata.batch.eq(batch)].sort_values('metadata_row')
        if rows.metadata_row.tolist() != list(range(1, 1001)):
            raise ValueError('Metadata coverage mismatch')
        with lzma.open(folders[0] / f'{batch}.xz', 'rb') as stream:
            for row in rows.itertuples():
                payload = stream.read(6881280)
                if len(payload) != 6881280:
                    raise ValueError('Truncated image')
                key = hashlib.sha256(payload).hexdigest()
                if key not in cache:
                    cache[key] = features_from_image(np.frombuffer(payload, dtype='<i2').reshape(480,7168))
                records.append(dict(sample_id=row.sample_id, batch=batch, raw_sha256=key, **cache[key]))
                if row.metadata_row % 250 == 0:
                    print(f'{batch}: extracted {row.metadata_row}/1000', flush=True)
            if stream.read(1):
                raise ValueError('Extra image bytes')
    frame = pd.DataFrame(records)
    if len(frame) != 7000 or not frame.sample_id.is_unique or not np.isfinite(frame[FEATURES]).all().all():
        raise ValueError('Feature coverage/finite-value check failed')
    frame.to_csv(ROOT / 'data/features_experimental.csv', index=False)
    print(f'Saved 7000 feature rows, {len(cache)} unique raw image hashes.')


if __name__ == '__main__':
    main()
