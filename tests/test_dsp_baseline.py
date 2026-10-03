"""Run with python -m unittest -v test_dsp_baseline.py."""
import inspect
import unittest
import numpy as np
import pandas as pd
import dsp_baseline as dsp


class DSPTests(unittest.TestCase):
    def test_analytic_envelope_dc_and_fixed_roi(self):
        image = np.zeros(dsp.SHAPE)
        # Integer number of sinusoid periods: analytic envelope should be constant.
        wave = 5 * np.sin(2 * np.pi * np.arange(2000) / 100)
        image[:, 1100:3100] = wave + np.arange(480)[:, None]
        roi, envelope, line = dsp.process_image(image)
        np.testing.assert_allclose(envelope, 5., atol=1e-10)
        baseline = dsp.image_scores(image)
        image[:, :1100] = 1e9
        image[:, 3100:] = -1e9
        image[:, 1100:3100] += np.arange(480)[:, None] * 2
        changed = dsp.image_scores(image)
        np.testing.assert_allclose(list(baseline.values()), list(changed.values()), atol=1e-10)
        self.assertEqual(list(inspect.signature(dsp.process_image).parameters), ['image'])
        self.assertEqual(list(inspect.signature(dsp.image_scores).parameters), ['image'])

    def test_full_window_spatial_mean(self):
        line = np.array([0., 0., 9., 0., 0.])
        np.testing.assert_array_equal(dsp.smooth_line(line, 3), [3., 3., 3.])
        self.assertEqual(dsp.smooth_line(line, 1).max(), 9.)

    def test_ties_and_metrics(self):
        result = dsp.metrics([0, 0, 1, 1], [0., 1., 1., 2.], [0, 1, 1, 1])
        self.assertEqual([result[k] for k in ['TP', 'TN', 'FP', 'FN']], [2, 1, 1, 0])
        self.assertAlmostEqual(result['ROC_AUC'], .875)
        self.assertAlmostEqual(result['balanced_accuracy'], .75)
        self.assertAlmostEqual(result['F1'], .8)

    def test_threshold_search_against_brute_force(self):
        rng = np.random.default_rng(9)
        validation = pd.DataFrame(dict(sample_id=range(40), has_flaw=[0, 1]*20, partition='validation'))
        for w in dsp.WINDOWS:
            validation[f'score_w{w}'] = rng.integers(0, 10, 40).astype(float)
        chosen, candidates = dsp.select_parameters(validation)
        for candidate in candidates:
            scores = validation[f'score_w{candidate["window"]}'].to_numpy()
            thresholds = np.r_[np.unique(scores), np.nextafter(scores.max(), np.inf)]
            expected = max((dsp.metrics(validation.has_flaw, scores, scores >= threshold)['balanced_accuracy'],
                            threshold) for threshold in thresholds)
            self.assertEqual((candidate['validation_balanced_accuracy'], candidate['threshold']), expected)

    def test_test_labels_scores_and_metadata_cannot_change_tuning(self):
        manifest = pd.DataFrame(dict(sample_id=list('abcdef'), has_flaw=[0, 1, 0, 1, 0, 1],
            partition=['validation']*4 + ['test']*2))
        scores = pd.DataFrame({'sample_id': list('abcdef'), **{f'score_w{w}': [0., 2., 1., 3., 4., 5.] for w in dsp.WINDOWS}})
        before = dsp.tune_scheme(manifest, scores)
        manifest.loc[manifest.partition.eq('test'), 'has_flaw'] = [1, 0]
        manifest['depth'] = 999999
        manifest['location'] = -999999
        scores.loc[scores.sample_id.isin(['e', 'f']), [f'score_w{w}' for w in dsp.WINDOWS]] = -1e20
        self.assertEqual(before, dsp.tune_scheme(manifest, scores))
        merged = manifest.merge(scores, on='sample_id')
        with self.assertRaises(ValueError):
            dsp.select_parameters(merged)


if __name__ == '__main__':
    unittest.main()
