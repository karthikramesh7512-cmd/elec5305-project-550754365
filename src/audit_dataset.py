"""Audit NDT metadata only. Requires pandas; no training or splitting.

Meanings follow the repository README. Physical units and source identity
are undocumented. Raw numeric values and the original type label are retained.
"""
from argparse import ArgumentParser
from pathlib import Path
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
NUMERIC = ['has_flaw', 'augmentation_amount', 'depth', 'location',
           'flaw_size', 'index_line']


def locate_dataset():
    folders = sorted({p.parent for p in PROJECT_DIR.rglob('batch_*.txt')
                      if p.parent.name == 'datasets'})
    if len(folders) != 1:
        raise ValueError(f'Found {folders}; specify --dataset-dir explicitly.')
    return folders[0]


def load_metadata(dataset_dir):
    files = sorted(dataset_dir.glob('batch_*.txt'))
    if not files:
        raise ValueError(f'No batch metadata in {dataset_dir}')
    frames = []
    for path in files:
        rows = [line.split() for line in path.read_text().splitlines()]
        widths = {len(row) for row in rows}
        if widths not in ({6}, {7}):
            raise ValueError(f'{path.name}: unexpected row widths {widths}')
        width = next(iter(widths))
        frame = pd.DataFrame(rows, columns=NUMERIC + (['flaw_type_raw'] if width == 7 else []))
        for column in NUMERIC:
            frame[column] = pd.to_numeric(frame[column], errors='raise')
        if not frame.has_flaw.isin([0, 1]).all():
            raise ValueError(f'{path.name}: invalid flaw indicator')
        if not frame.index_line.eq(1).all():
            raise ValueError(f'{path.name}: index_line is not constant 1')
        frame['has_flaw'] = frame.has_flaw.astype(int)
        frame['index_line'] = frame.index_line.astype(int)
        if width == 6:
            frame['flaw_type_raw'] = pd.NA
        frame['flaw_type_raw'] = frame.flaw_type_raw.astype('string')
        frame['flaw_type'] = frame.flaw_type_raw.replace('-', pd.NA)
        frame['flaw_label'] = frame.has_flaw.map({0: 'no_flaw', 1: 'flaw'})
        frame['batch'] = path.stem
        frame['metadata_file'] = path.name
        frame['metadata_row'] = range(1, len(frame) + 1)
        frame['sample_id'] = frame.batch + ':' + frame.metadata_row.astype(str)
        frame['metadata_column_count'] = width
        simulated = 200 <= int(path.stem.removeprefix('batch_')) <= 299
        frame['batch_origin'] = 'simulated' if simulated else 'experimental'
        frame['batch_origin_basis'] = ('README: 2xx simulated' if simulated else
            'inferred: non-2xx; experimental not explicitly confirmed')
        # Provisional grouping label, NOT a verified physical source identifier.
        frame['source_flaw'] = frame.flaw_type.where(frame.has_flaw.eq(1))
        frame['source_flaw_status'] = 'unknown_missing_label'
        frame.loc[frame.source_flaw.notna(), 'source_flaw_status'] = 'provisional_from_flaw_type_label'
        frame.loc[frame.has_flaw.eq(0), 'source_flaw_status'] = 'not_applicable_no_flaw'
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def print_audit(df, dataset_dir):
    print(f'Dataset folder: {dataset_dir}')
    print('''\nMetadata fields in file order (README):
1. has_flaw: 1 = flaw, 0 = no flaw
2. augmentation_amount: README says 0.4-1; exact operation unspecified
3. depth: flaw depth; units unspecified
4. location: flaw location along scan axis; units unspecified
5. flaw_size: ORIGINAL flaw size; units unspecified
6. index_line: always 1; not a sample/source identifier
7. flaw_type_raw: README's Flaw type; absent in six-column files
Raw zeros and raw '-' labels are preserved. Normalized absent labels are null.''')
    print(f'\nTotal samples: {len(df):,}')
    print('\nFlaw vs no-flaw counts:\n' + df.flaw_label.value_counts().to_string())
    print('\nUnique source flaws (provisional labels only):')
    print(sorted(df.source_flaw.dropna().unique().tolist()))
    print('\nUnique flaw types (README labels, not a documented morphology taxonomy):')
    print(sorted(df.flaw_type.dropna().unique().tolist()))
    print('\nSamples per batch:')
    print(df.groupby(['batch', 'batch_origin', 'metadata_column_count']).size().rename('samples').to_string())
    source_groups = df.source_flaw.fillna('[unknown source: flaw]')
    source_groups = source_groups.mask(df.has_flaw.eq(0), '[no flaw]')
    print('\nSamples per source flaw (provisional; missing groups included):')
    print(source_groups.value_counts().sort_index().to_string())
    print('\nSource identification status:\n' + df.source_flaw_status.value_counts().to_string())
    print('\nSize/depth consistency per recorded source label:')
    print(df.loc[df.source_flaw.notna()].groupby('source_flaw').agg(
        sizes=('flaw_size', lambda x: sorted(x.unique())),
        depths=('depth', lambda x: sorted(x.unique())),
        batches=('batch', 'nunique')).to_string())
    print('\nObserved numeric ranges by origin and flaw label:')
    print(df.groupby(['batch_origin', 'flaw_label'])[NUMERIC[1:5]].agg(['min', 'max']).to_string())
    print('''\nReliability: source_flaw is a provisional copy of the recorded type label.
Physical source identity is NOT verified by the README or metadata.
Simulated rows lack a type/source label; size/depth are not source IDs.
Non-2xx origin is inferred as experimental; only 2xx simulation is explicit.''')


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-dir', type=Path)
    parser.add_argument('--output', type=Path, default=PROJECT_DIR / 'data' / 'metadata.csv')
    args = parser.parse_args()
    dataset_dir = args.dataset_dir.resolve() if args.dataset_dir else locate_dataset()
    metadata = load_metadata(dataset_dir)
    print_audit(metadata, dataset_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    metadata.to_csv(args.output, index=False)
    print(f'\nSaved combined metadata: {args.output.resolve()}')


if __name__ == '__main__':
    main()
