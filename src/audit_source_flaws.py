"""Summarize existing metadata.csv; never modify sample metadata or make splits."""
from pathlib import Path
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PAPER = 'https://doi.org/10.1007/s10921-021-00757-x'


def unique_json(series):
    return json.dumps(sorted(series.dropna().unique().tolist()))


def build_audit(metadata):
    # Existing CSV names: flaw_size = original_flaw_size; depth = flaw_depth.
    positive = metadata.loc[metadata.has_flaw.eq(1)].copy()
    if positive[['flaw_size', 'depth', 'batch', 'batch_origin']].isna().any().any():
        raise ValueError('Missing required positive-sample metadata')
    experimental = positive.loc[positive.batch_origin.eq('experimental')]
    simulated = positive.loc[positive.batch_origin.eq('simulated')]
    rows = []
    for label, group in experimental.dropna(subset=['flaw_type']).groupby('flaw_type'):
        sizes = sorted(group.flaw_size.unique().tolist())
        rows.append(dict(
            source_label=label, domain='experimental',
            original_size=sizes[0] if len(sizes) == 1 else pd.NA,
            n_samples=len(group), batches=';'.join(sorted(group.batch.unique())),
            original_flaw_size_values=unique_json(group.flaw_size),
            flaw_depth_values=unique_json(group.depth),
            n_unique_original_sizes=len(sizes),
            label_basis='direct metadata flaw_type label; physical identity inferred',
            paper_supported_size_mm=(sizes[0] if len(sizes) == 1 and sizes[0] in {2, 3, 6, 17, 26} else pd.NA),
            candidate_notch_size_mm=pd.NA,
            mapping_status='scanned size inventory consistent with paper; P41 codebook not supplied',
            paper_reference=PAPER,
        ))
    # Explicit hypothesis only: do not replace raw values or claim verified mm.
    candidates = {0.001: 1, 1.001: 2, 2.001: 3, 3.001: 4, 4.001: 5, 5.001: 6}
    for size, group in simulated.groupby('flaw_size'):
        rows.append(dict(
            source_label=f'sim_raw_{size:g}', domain='simulated',
            original_size=size, n_samples=len(group),
            batches=';'.join(sorted(group.batch.unique())),
            original_flaw_size_values=unique_json(group.flaw_size),
            flaw_depth_values=unique_json(group.depth), n_unique_original_sizes=1,
            label_basis='audit-generated group name from raw size; not an encoded source label',
            paper_supported_size_mm=pd.NA,
            candidate_notch_size_mm=candidates.get(size, pd.NA),
            mapping_status='unverified ordered mapping; paper confirms six heights but not raw encoding',
            paper_reference=PAPER,
        ))
    audit = pd.DataFrame(rows)
    omitted = len(experimental.loc[experimental.flaw_type.isna()])
    if int(audit.n_samples.sum()) + omitted != len(positive):
        raise ValueError('Unaccounted positive samples/domain')
    return audit, omitted


def main():
    metadata = pd.read_csv(ROOT / 'data' / 'metadata.csv')
    audit, omitted = build_audit(metadata)
    output = ROOT / 'data' / 'source_flaw_audit.csv'
    audit.to_csv(output, index=False)
    print(audit[['source_label', 'domain', 'original_size', 'n_samples', 'batches']].to_string(index=False))
    experimental = audit.loc[audit.domain.eq('experimental')]
    print('\nExperimental unique sizes/depths:')
    print(experimental[['source_label', 'original_flaw_size_values', 'flaw_depth_values']].to_string(index=False))
    print('\nEvery P41 label has one original size:',
          experimental.loc[experimental.source_label.str.startswith('P41'), 'n_unique_original_sizes'].eq(1).all())
    print('Experimental positive rows without a label:', omitted)
    print('Simulated raw size categories:', len(audit.loc[audit.domain.eq('simulated')]))
    print('Simulated mm mapping is an explicit hypothesis, NOT verified metadata.')
    print('CSV flaw_size and depth are reported as original_flaw_size and flaw_depth.')
    print(f'Saved: {output}')


if __name__ == '__main__':
    main()
