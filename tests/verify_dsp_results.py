"""Independently check saved manifests, selection provenance, and predictions."""
import json
import numpy as np
import pandas as pd
import dsp_baseline as dsp


def main():
    ordinary = pd.read_csv(dsp.ROOT / 'data/splits/ordinary_split.csv')
    holdout = pd.read_csv(dsp.ROOT / 'data/splits/source_holdout_folds.csv')
    scores = pd.read_csv(dsp.OUT / 'candidate_scores.csv')
    predictions = pd.read_csv(dsp.OUT / 'per_sample_results.csv')
    selected = pd.read_csv(dsp.OUT / 'selected_parameters.csv')
    tuning = pd.read_csv(dsp.OUT / 'tuning_sample_ids.csv')
    summary = pd.read_csv(dsp.OUT / 'dsp_summary.csv')
    checks = json.loads((dsp.OUT / 'checks.json').read_text())
    for path, expected in checks['split_sha256'].items():
        dsp.require(dsp.digest(dsp.ROOT / path.replace(chr(92), "/")) == expected, f'Split modified: {path}')
    dsp.require(len(predictions) == 49000, 'Expected 7000 ordinary + 42000 fold predictions')
    dsp.require(len(summary) == 7 and len(selected) == 7, 'Expected seven evaluations')
    for scheme, fold, manifest in [('ordinary', 0, ordinary)] + [
            ('source_holdout', int(f), m) for f, m in holdout.groupby('fold')]:
        key = lambda df: df.scheme.eq(scheme) & df.fold.eq(fold)
        result = predictions.loc[key(predictions)]
        pd.testing.assert_frame_equal(
            result[['sample_id', 'partition']].sort_values('sample_id').reset_index(drop=True),
            manifest[['sample_id', 'partition']].sort_values('sample_id').reset_index(drop=True))
        dsp.require(result.sample_id.is_unique, 'Duplicate predictions within evaluation')
        actual_tuning = tuning.loc[key(tuning)]
        expected_validation = set(manifest.loc[manifest.partition.eq('validation'), 'sample_id'])
        expected_test = set(manifest.loc[manifest.partition.eq('test'), 'sample_id'])
        dsp.require(actual_tuning.sample_id.is_unique and set(actual_tuning.sample_id) == expected_validation,
                    'Tuning IDs differ from validation IDs')
        dsp.require(not set(actual_tuning.sample_id) & expected_test, 'Test used for tuning')
        choice = selected.loc[key(selected)].iloc[0]
        # Check score/prediction arithmetic; do not reselect parameters against tests.
        joined = result.merge(scores, on='sample_id', validate='one_to_one')
        np.testing.assert_allclose(joined.dsp_score, joined[f'score_w{int(choice.window)}'], rtol=1e-14)
        np.testing.assert_array_equal(joined.predicted_label, (joined.dsp_score >= choice.threshold).astype(int))
        test = result.loc[result.partition.eq('test')]
        dsp.require(set(test.sample_id) == expected_test, 'Incomplete test coverage')
        measured = dsp.metrics(test.true_label, test.dsp_score, test.predicted_label)
        saved = summary.loc[key(summary)].iloc[0]
        for field, value in measured.items():
            np.testing.assert_allclose(saved[field], value, rtol=1e-13, atol=1e-15)
    dsp.require(len(predictions.loc[predictions.partition.eq('test')]) == 8050,
                'Expected 1050 ordinary test + 7000 source-holdout test predictions')
    (dsp.OUT / 'artifact_verification.json').write_text(json.dumps(dict(
        passed=True, evaluations=7, prediction_rows=49000, test_prediction_rows=8050,
        tuning_ids_match_validation=True, test_tuning_overlap=False,
        split_hashes_unchanged=True, saved_metrics_match_predictions=True), indent=2))
    print('PASS: seven evaluations; 49000 predictions; 8050 test predictions;')
    print('validation-only tuning IDs; intact split hashes; all saved metrics verified.')


if __name__ == '__main__':
    main()
