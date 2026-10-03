"""Finish the SVM report from saved outputs. No extraction, fitting or tuning."""
import json
import pandas as pd
import numpy as np
import svm_baseline as svm
from extract_features import FEATURES
OUT=svm.OUT


def md(frame):
    def fmt(x):
        return f'{x:.6g}' if isinstance(x,float) else str(x)
    return '\n'.join(['| '+' | '.join(map(str,frame.columns))+' |',
        '| '+' | '.join(['---']*len(frame.columns))+' |']+
        ['| '+' | '.join(fmt(x) for x in row)+' |' for row in frame.itertuples(index=False,name=None)])


def main():
    checks=json.loads((OUT/'checks.json').read_text())
    for name,value in checks['protected_sha256'].items():
        svm.require(svm.sha(svm.ROOT/name)==value,f'Protected artifact changed: {name}')
    frozen=[OUT/'svm_summary.csv',OUT/'per_sample_results.csv',OUT/'selected_parameters.csv',
            svm.ROOT/'data/features_experimental.csv']+list(OUT.glob('*_pipeline.joblib'))
    before={str(p.relative_to(svm.ROOT)):svm.sha(p) for p in frozen}
    summary=pd.read_csv(OUT/'svm_summary.csv')
    dsp=pd.read_csv(svm.ROOT/'results/dsp_baseline/dsp_summary.csv')
    comparison=summary[['scheme','fold','test_source','flaw_size','recall','balanced_accuracy','ROC_AUC']].merge(
        dsp[['scheme','fold','recall','balanced_accuracy','ROC_AUC']],on=['scheme','fold'],suffixes=('_svm','_dsp'),validate='one_to_one')
    comparison.insert(0,'evaluation',['Ordinary' if r.scheme=='ordinary' else f'Source holdout {r.fold}' for r in comparison.itertuples()])
    comparison=comparison.rename(columns={'test_source':'held_out_source','recall_dsp':'DSP_recall','recall_svm':'SVM_recall',
        'balanced_accuracy_dsp':'DSP_balanced_accuracy','balanced_accuracy_svm':'SVM_balanced_accuracy',
        'ROC_AUC_dsp':'DSP_ROC_AUC','ROC_AUC_svm':'SVM_ROC_AUC'})
    comparison=comparison[['evaluation','held_out_source','flaw_size','DSP_recall','SVM_recall',
        'DSP_balanced_accuracy','SVM_balanced_accuracy','DSP_ROC_AUC','SVM_ROC_AUC']]
    comparison.to_csv(OUT/'dsp_vs_svm.csv',index=False)
    audit=pd.read_csv(OUT/'feature_audit.csv')
    corr=pd.read_csv(OUT/'feature_correlation.csv',index_col=0)
    notes=[]
    for feature in audit.feature:
        partners=[f'{other}: r={corr.loc[feature,other]:.6f}' for other in FEATURES
                  if other!=feature and abs(corr.loc[feature,other])>=.95]
        notes.append('; '.join(partners) if partners else 'No pair with |r| >= 0.95')
    audit['redundancy_observations']=notes
    audit.to_csv(OUT/'feature_audit.csv',index=False)
    coeff=pd.read_csv(OUT/'linear_feature_coefficients.csv')
    selected=coeff.loc[coeff.selected_model_kernel.eq('linear')].copy()
    selected['absolute_weight_rank']=selected.groupby(['scheme','fold']).absolute_coefficient.rank(method='min',ascending=False).astype(int)
    selected.to_csv(OUT/'selected_linear_coefficients.csv',index=False)
    top=selected.loc[selected.absolute_weight_rank.le(3)].sort_values(['scheme','fold','absolute_weight_rank'])
    top.to_csv(OUT/'largest_linear_weights.csv',index=False)
    wide=selected.assign(evaluation=lambda x: np.where(x.scheme.eq('ordinary'),'Ordinary','Fold '+x.fold.astype(str))).pivot(
        index='feature',columns='evaluation',values='standardized_coefficient').reindex(FEATURES).reset_index()
    p=pd.read_csv(OUT/'per_sample_results.csv')
    errors=p.loc[p.partition.eq('test') & p.has_flaw.ne(p.predicted_label)].copy()
    f=pd.read_csv(svm.ROOT/'data/features_experimental.csv')
    errors=errors.merge(f[['sample_id','raw_sha256']],on='sample_id',validate='many_to_one')
    errors.to_csv(OUT/'test_error_cases.csv',index=False)
    row=summary.loc[summary.test_source.eq('P41_01')].iloc[0]
    svm.require(row.TP==540 and row.FN==7 and np.isclose(row.recall,540/547),'P41_01 count mismatch')
    for _,r in summary.loc[summary.scheme.eq('source_holdout')].iterrows():
        fig,ax=svm.plt.subplots(figsize=(5,4),layout='constrained')
        svm.confusion(ax,r,f'Fold {r.fold}: {r.test_source}')
        fig.savefig(OUT/f'confusion_fold_{r.fold}_{r.test_source}.png',dpi=150);svm.plt.close(fig)
    trials=pd.read_csv(OUT/'validation_grid.csv')
    kernel_comparison=trials.groupby(['scheme','fold','kernel']).validation_balanced_accuracy.max().unstack().reset_index()
    kernel_comparison.to_csv(OUT/'kernel_validation_comparison.csv',index=False)
    params=pd.read_csv(OUT/'selected_parameters.csv').drop(columns='candidate')
    params.loc[params.kernel.eq('linear'),'gamma']='not applicable'
    metrics=summary[['test_source','TP','TN','FP','FN','accuracy','recall','specificity','precision','F1',
                     'false_positive_rate','balanced_accuracy','ROC_AUC','PR_AUC']].copy()
    metrics.loc[metrics.test_source.eq('mixed'),'test_source']='Ordinary'
    source=pd.read_csv(OUT/'per_source_comparison.csv')
    definitions=pd.DataFrame([
        ('max_envelope','Maximum Hilbert-envelope amplitude in fixed ROI'),
        ('p95_envelope','95th percentile of all ROI envelope values (linear interpolation)'),
        ('mean_envelope','Mean of all ROI envelope values'),
        ('rf_rms','RMS of DC-removed RF ROI'),
        ('rf_energy','Sum of squared DC-removed RF samples; 960000 * RMS squared'),
        ('envelope_energy','Sum of squared envelope samples'),
        ('crest_factor','Maximum absolute DC-removed RF / RF RMS; zero for zero signal'),
        ('strong_peak_count','Spatial line-response local peaks: height >= 0.5 max, prominence >= 0.1 max, distance >= 3 scan indices; endpoints eligible'),
        ('spatial_persistence','Longest contiguous scan run with line response >= 0.5 max'),
        ('strongest_response_width','Interpolated half-maximum width of run containing global spatial maximum; clipped at image edge'),
        ('line_peak_std','Population SD over 480 per-A-scan maximum envelope values'),
        ('line_peak_mean','Mean over 480 per-A-scan maximum envelope values')],columns=['Feature','Definition'])
    report=f'''# Engineered-feature SVM: completed experiment

## Scope and main result

Only experimental batches 013–019 and the existing ordinary/six source-holdout
manifests were used. Feature extraction and saved model fits were reused for this
finalization; no real-data model was refitted and no test-set retuning was performed.
DSP Baseline V1 and all split files remain unchanged. No CNN is implemented.

**All 3,558 no-flaw experimental images are exact duplicates.** Therefore,
specificity cannot be interpreted as generalisation to novel negative examples.
Ordinary image-level performance is mainly a reference result. Source-flaw holdout
is the main generalisation result, limited to these six sources on a shared
experimental background; it does not establish new-specimen generalisation.

P41_01 has **540 TP / 7 FN**, recall **540/547 = 98.7202925%**. P41_02 and all
other held-out sources have 100% recall. These improve on DSP V1 for P41_01 and
P41_02; no held-out source worsens. There are no false positives on the duplicated
negative background. The seven P41_01 errors cover {errors.raw_sha256.nunique()}
distinct raw-image hashes, so even the false-negative rows are not all independent.

## Selected models and validation comparison

{md(params)}

Candidates: linear C in [0.01,0.1,1,10,100]; RBF C in [0.1,1,10,100] with
gamma in [scale,0.001,0.01,0.1,1]. Validation balanced accuracy alone selects.
Ties follow fixed grid order: linear first, ascending C, then listed gamma order.
Thus a winning linear model need not be uniquely best. The RBF fold has no
direct linear coefficient interpretation. Gamma is unused for linear models.

Best validation balanced accuracy per kernel:

{md(kernel_comparison)}

## Complete test metrics

All rates below are fractions. Sensitivity equals recall. PR-AUC is trapezoidal
area under the precision-recall curve of SVC decision scores; average precision
is saved separately and is also 1.0 in each evaluation. Scores are uncalibrated
decision values, not probabilities. SVC's original decision rule was retained.

{md(metrics)}

## DSP V1 versus SVM

{md(comparison)}

Per-source recall differences (SVM minus DSP; fractions):

{md(source.loc[source.scheme.eq('source_holdout'),['source_flaw','flaw_size','svm_recall','dsp_recall','recall_difference']])}

## Feature definitions and audit

Fixed ROI `image[:,1100:3100]`, float64, independent DC removal per A-scan,
Hilbert envelope along axis 1. Feature extraction receives only image pixels.
Labels, source, size, depth, location, IDs and hashes are excluded from model X;
X uses an explicit allowlist of exactly 12 feature columns.

{md(definitions)}

Population SD uses ddof=0. Near-constant means one value occupies >=99% of rows,
or SD <= 1e-6 * max(abs(mean),1), including constants. No feature meets those
criteria globally. All features are constant within the duplicated no-flaw class.
No features were removed. Full-data audit/correlation statistics are descriptive
only: they did not affect the feature list, scaling, model grid or selection.

{md(audit[['feature','mean','std','minimum','maximum']])}

Per-feature correlation partners are in `feature_audit.csv`; all pairs with
absolute Pearson r >=0.95 are in `redundant_features.csv`. Strong redundancy:
RF energy vs envelope energy r≈1; RMS vs RF energy r≈0.999016; persistence vs
strongest-response width r≈0.999358; envelope p95 vs mean r≈0.996867.
RF energy is exactly 960000 * RF RMS squared. Envelope energy is nearly twice
RF energy for these centered RF traces. Keeping correlated features changes the
geometry seen by the SVM; coefficients are consequently not independent effects.

## Standardised linear coefficients

The signed weights below multiply training-standardised features; positive values
increase the flaw-class decision value, with other standardised inputs fixed.
They are **not causal feature importances**. Coefficient magnitudes and ranks
for every selected linear model are in `selected_linear_coefficients.csv`.

{md(wide)}

The largest absolute weights are usually persistence and strongest-response width,
both negative. Fold P41_01 instead gives its largest magnitude to strong-peak
count (-0.269885), followed by persistence (-0.255439) and width (-0.250226).
In the ordinary model the three largest are persistence (-0.491512), width
(-0.484246), and strong-peak count (-0.222837).

The repeated negative background has persistence 88 scan samples, strongest width
88.0712 and 38 strong peaks. Positive median persistence is 8–15 and peak counts
1–27. Because the threshold is relative to each image's maximum, stronger local
flaw echoes can concentrate the above-half-maximum response. This is a supported
feature observation, not physical crack-width measurement or a causal claim.
RBF fold P41_06_notch has no direct linear coefficients. The older
`linear_feature_coefficients.csv` additionally records its best *unselected*
linear candidate; that candidate must not be interpreted as the selected RBF.

## Failure cases and warnings

{md(errors[['sample_id','source_label','decision_score','predicted_label']])}

All seven missed P41_01 positives have negative decision values. One is near the
boundary (-0.000586); the others are approximately -0.126 to -0.142. No adjustment
was made after seeing them. ROC-AUC and PR-AUC remain 1.0 because positive scores
still rank above negative scores even when the zero decision boundary misses some.
Ranking performance and recall at an operating point answer different questions.

The negative duplication makes specificity an observation about one reused
background, not a performance estimate for diverse negative acquisitions.
Ordinary perfect performance must be read as a reference result, with duplicated
content and related augmented positives across partitions. The six held-out-source
results are more informative but retain the shared-background limitation.

## Figures

- `dsp_vs_svm_balanced_accuracy.png` and `dsp_vs_svm_recall.png`.
- `ordinary_confusion_matrix.png`.
- `source_holdout_confusion_matrices.png` plus six `confusion_fold_*.png` files.
- `feature_correlation.png`, `feature_distributions.png` (ordinary train/validation).
- `pca_diagnostic.png`: diagnostic only. PCA and its scaler fit ordinary training
  rows only; all partitions are transformed for display. PCA is not a classifier input.

## Verification and reproducibility

Run `python -m unittest discover -v` for feature/DSP unit and SVM provenance tests.
`test_results.txt` records the final test execution. Checks cover:

- Scaler fitted on training rows only: saved n_samples_seen, means and variances
  agree with training data.
- SVC fitted on training only: fit-call tests, logged IDs, saved fit shape and
  support vectors matching training-scaled rows.
- Validation labels used for hyperparameter selection, never scaler/SVC fitting.
- No test IDs in fitting or selection; test labels never enter model selection.
- Held-out sources absent from training; exactly one prediction per expected test
  sample within each protocol/fold (8,050 total test predictions across protocols).
- Frozen DSP outputs/splits retain SHA-256 hashes; saved SVM models, features,
  summary and predictions remain unchanged during finalization.

Pipeline: `StandardScaler()` then `SVC()`. No train+validation refit, threshold
retuning, class weighting, oversampling, feature removal, or PCA classification.
The pre-existing duplicate negatives were retained as required by the fixed splits.
Dependencies are installed in project-local `.python_deps`; sklearn version is
{svm.sklearn.__version__}. `extract_features.py` and `svm_baseline.py` reproduce the
original extraction/evaluation; `finalize_svm_report.py` updates reporting only.
'''
    (OUT/'svm_report.md').write_text(report,encoding='utf-8')
    after={str(p.relative_to(svm.ROOT)):svm.sha(p) for p in frozen}
    svm.require(before==after,'Features, fitted models or metric results changed')
    for name,value in checks['protected_sha256'].items():
        svm.require(svm.sha(svm.ROOT/name)==value,'Protected artifact changed')
    (OUT/'finalization_checks.json').write_text(json.dumps(dict(all_preserved=True,
        svm_frozen_artifact_sha256=before,p41_01_TP=540,p41_01_FN=7,recall=540/547,
        no_extraction_or_fitting=True),indent=2))
    print('Final report and audits saved; existing models, metrics, predictions, DSP and splits preserved.')


if __name__=='__main__':main()
