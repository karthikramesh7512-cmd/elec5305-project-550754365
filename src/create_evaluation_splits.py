"""Create and validate experimental-only evaluation manifests; no DSP or ML.

Run: python create_evaluation_splits.py
Sample IDs refer to metadata rows (one-based), not hashes of image contents.
Ordinary splitting allows sources to overlap; source holdout isolates P41 labels.
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SEED = 42
BATCHES = [f'batch_{number:03d}' for number in range(13, 20)]
SOURCES = ['P41_01', 'P41_02', 'P41_03', 'P41_04', 'P41_05', 'P41_06_notch']
EXPECTED_COUNTS = [547, 543, 621, 558, 574, 599]
EXPECTED_SIZES = [2, 3, 26, 6, 17, 6]
PARTITIONS = ['train', 'validation', 'test']
COLUMNS = ['sample_id', 'batch', 'has_flaw', 'source_label', 'flaw_size',
           'metadata_file', 'metadata_row']


def require(condition, message):
    """Checks remain enabled even when Python runs with -O."""
    if not condition:
        raise ValueError(message)


def load_experimental(path):
    metadata = pd.read_csv(path)
    data = metadata.loc[metadata.batch.isin(BATCHES)].copy()
    data = data.sort_values(['batch', 'metadata_row']).reset_index(drop=True)
    data['source_label'] = data.flaw_type.where(data.has_flaw.eq(1))
    require(set(data.batch) == set(BATCHES), 'Missing experimental batch')
    require(data.batch_origin.eq('experimental').all(), 'Unexpected domain')
    require(len(data) == 7000 and data.sample_id.is_unique, 'Sample count/ID mismatch')
    require(data.groupby('batch').size().eq(1000).all(), 'Batch counts changed')
    require(data.has_flaw.isin([0, 1]).all(), 'Invalid class label')
    require(data.has_flaw.sum() == 3442, 'Expected 3442 positives / 3558 negatives')
    positive = data.loc[data.has_flaw.eq(1)]
    require(positive.source_label.notna().all(), 'Positive sample has no source')
    require(set(positive.source_label) == set(SOURCES), 'Unexpected source labels')
    for source, count, size in zip(SOURCES, EXPECTED_COUNTS, EXPECTED_SIZES):
        group = positive.loc[positive.source_label.eq(source)]
        require(len(group) == count and group.flaw_size.eq(size).all(),
                f'Source inventory mismatch: {source}')
    return data[COLUMNS].copy()


def ordinary_split(data):
    rng = np.random.default_rng(SEED)
    result = data.copy()
    result['partition'] = ''
    totals = np.array([4900, 1050, 1050])
    # Round positive quotas for train/validation; preserve exact total sizes.
    n_positive = int(data.has_flaw.sum())
    positive_counts = np.array([round(n_positive * .70), round(n_positive * .15), 0])
    positive_counts[2] = n_positive - positive_counts[:2].sum()
    for label, quotas in [(1, positive_counts), (0, totals - positive_counts)]:
        indices = rng.permutation(data.index[data.has_flaw.eq(label)].to_numpy())
        offset = 0
        for partition, count in zip(PARTITIONS, quotas):
            result.loc[indices[offset:offset + count], 'partition'] = partition
            offset += count
    return result


def assign_negative_groups(data):
    rng = np.random.default_rng(SEED + 1)
    groups = pd.Series(pd.NA, index=data.index, dtype='Int64')
    cursor = 0
    for batch in BATCHES:
        indices = data.index[data.batch.eq(batch) & data.has_flaw.eq(0)].to_numpy()
        indices = rng.permutation(indices)
        # Continuing the round-robin across batches gives exactly 593 per group,
        # and each batch contributes counts differing by at most one.
        groups.loc[indices] = (np.arange(len(indices)) + cursor) % 6 + 1
        cursor = (cursor + len(indices)) % 6
    return groups


def source_holdout(data):
    groups = assign_negative_groups(data)
    folds = []
    for index, test_source in enumerate(SOURCES):
        fold = index + 1
        validation_group = (index + 1) % 6 + 1
        validation_source = SOURCES[validation_group - 1]
        frame = data.copy()
        frame['fold'] = fold
        frame['partition'] = 'train'
        frame['negative_group'] = groups
        frame['test_source'] = test_source
        frame['validation_source'] = validation_source
        frame.loc[frame.source_label.eq(test_source) | groups.eq(fold).fillna(False), 'partition'] = 'test'
        frame.loc[frame.source_label.eq(validation_source) | groups.eq(validation_group).fillna(False), 'partition'] = 'validation'
        folds.append(frame)
    return pd.concat(folds, ignore_index=True)


def class_counts(frame):
    table = pd.crosstab(frame.partition, frame.has_flaw).reindex(
        index=PARTITIONS, columns=[0, 1], fill_value=0)
    table.columns = ['no_flaw', 'flaw']
    table['total'] = table.sum(axis=1)
    return table


def validate(data, ordinary, holdout):
    expected_ids = set(data.sample_id)

    def check_coverage(frame):
        require(len(frame) == len(data) and frame.sample_id.is_unique,
                'Duplicate sample or wrong row count within split/fold')
        require(set(frame.sample_id) == expected_ids, 'Incomplete sample coverage')
        require(set(frame.partition) == set(PARTITIONS), 'Invalid partitions')
        actual = frame[COLUMNS].sort_values('sample_id').reset_index(drop=True)
        expected = data.sort_values('sample_id').reset_index(drop=True)
        pd.testing.assert_frame_equal(actual, expected, check_dtype=False)

    check_coverage(ordinary)
    counts = class_counts(ordinary)
    require(counts.total.tolist() == [4900, 1050, 1050], 'Ordinary ratios incorrect')
    require((counts.flaw - counts.total * data.has_flaw.mean()).abs().lt(1).all(),
            'Ordinary class quotas differ from proportional targets by >=1')
    require(set(holdout.fold) == set(range(1, 7)), 'Expected six folds')
    for fold, frame in holdout.groupby('fold'):
        check_coverage(frame)
        test_source = SOURCES[fold - 1]
        validation_source = SOURCES[fold % 6]
        for partition, sources in [
            ('test', {test_source}), ('validation', {validation_source}),
            ('train', set(SOURCES) - {test_source, validation_source}),
        ]:
            actual_sources = set(frame.loc[frame.partition.eq(partition) & frame.has_flaw.eq(1), 'source_label'])
            require(actual_sources == sources, f'Source leakage in fold {fold}/{partition}')
        require(frame.loc[frame.source_label.eq(test_source), 'partition'].eq('test').all(), 'Test source leaked')
        require(frame.loc[frame.source_label.eq(validation_source), 'partition'].eq('validation').all(), 'Validation source leaked')
        negatives = frame.loc[frame.has_flaw.eq(0)]
        require(negatives.negative_group.notna().all(), 'Negative group missing')
        expected_partition = negatives.negative_group.map(
            lambda group: 'test' if group == fold else 'validation' if group == fold % 6 + 1 else 'train')
        require(negatives.partition.eq(expected_partition).all(), 'Negative group leakage')
        require(class_counts(frame).no_flaw.tolist() == [2372, 593, 593], 'Negative counts incorrect')
    negative_rows = holdout.loc[holdout.has_flaw.eq(0)]
    require(negative_rows.groupby('sample_id').negative_group.nunique().eq(1).all(), 'Negative groups changed between folds')
    balance = pd.crosstab(negative_rows.loc[negative_rows.fold.eq(1), 'batch'],
                          negative_rows.loc[negative_rows.fold.eq(1), 'negative_group'])
    require((balance.max(axis=1) - balance.min(axis=1)).le(1).all(), 'Batch balance failed')
    require(balance.sum().eq(593).all(), 'Unequal negative groups')
    usage = pd.crosstab(holdout.sample_id, holdout.partition)
    require(usage.test.eq(1).all() and usage.validation.eq(1).all() and usage.train.eq(4).all(),
            'Each sample must be test once, validation once, train four times')
    return balance


def main():
    data = load_experimental(ROOT / 'data' / 'metadata.csv')
    ordinary = ordinary_split(data)
    holdout = source_holdout(data)
    balance = validate(data, ordinary, holdout)
    destination = ROOT / 'data' / 'splits'
    destination.mkdir(parents=True, exist_ok=True)
    ordinary.to_csv(destination / 'ordinary_split.csv', index=False)
    holdout.to_csv(destination / 'source_holdout_folds.csv', index=False)
    print(f'Seed: {SEED}; experimental batches 013-019 only')
    print('\nOrdinary image-level split:\n' + class_counts(ordinary).to_string())
    print('\nFixed negative groups per batch:\n' + balance.to_string())
    for fold, frame in holdout.groupby('fold'):
        print(f'\nFold {fold}: test={SOURCES[fold - 1]}, validation={SOURCES[fold % 6]}')
        print(class_counts(frame).to_string())
    print('\nPASS: coverage, source exclusion, sample disjointness, fixed negative groups,')
    print('batch balance, ordinary stratification, and once-per-sample test/validation rotation.')
    print(f'Saved manifests to {destination}')


if __name__ == '__main__':
    main()
