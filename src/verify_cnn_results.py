"""Independent saved-artifact verification. Never fits a model or rewrites results."""
from pathlib import Path
import json, hashlib
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, confusion_matrix

ROOT=Path(__file__).resolve().parents[1]

def main():
    out=ROOT/'results/dsp_cnn'
    config=json.loads((out/'experiment_config.json').read_text())
    for path,digest in config['protected_sha256'].items():
        assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==digest,path
    ordinary=pd.read_csv(ROOT/'data/splits/ordinary_split.csv')
    holdout=pd.read_csv(ROOT/'data/splits/source_holdout_folds.csv')
    pred=pd.read_csv(out/'per_sample_results.csv');summary=pd.read_csv(out/'cnn_summary.csv')
    selected=pd.read_csv(out/'checkpoint_selection.csv')
    assert len(summary)==len(selected)==21
    for row in summary.itertuples():
        manifest=ordinary if row.scheme=='ordinary' else holdout[holdout.fold.eq(row.fold)]
        test=manifest[manifest.partition.eq('test')].set_index('sample_id')
        saved=pred[pred.scheme.eq(row.scheme)&pred.fold.eq(row.fold)&pred.seed.eq(row.seed)].set_index('sample_id')
        assert saved.index.is_unique and set(saved.index)==set(test.index)
        np.testing.assert_array_equal(saved.has_flaw,test.loc[saved.index,'has_flaw'])
        np.testing.assert_array_equal(saved.prediction,(saved.logit>=0).astype(int))
        np.testing.assert_allclose(saved.probability,1/(1+np.exp(-saved.logit)),atol=1e-6)
        assert abs(balanced_accuracy_score(saved.has_flaw,saved.prediction)-row.balanced_accuracy)<1e-12
        assert abs(roc_auc_score(saved.has_flaw,saved.logit)-row.ROC_AUC)<1e-12
        tn,fp,fn,tp=confusion_matrix(saved.has_flaw,saved.prediction,labels=[0,1]).ravel()
        assert (tn,fp,fn,tp)==(row.TN,row.FP,row.FN,row.TP)
        stem=f'{row.scheme}_{row.fold}_seed_{row.seed}'
        fit_ids=pd.read_csv(out/f'{stem}_selection_ids.csv')
        assert set(fit_ids.sample_id)==set(manifest.loc[~manifest.partition.eq('test'),'sample_id'])
        assert not set(fit_ids.sample_id)&set(test.index)
        h=pd.read_csv(out/f'{stem}_history.csv')
        expected=h.sort_values(['validation_balanced_accuracy','validation_loss','epoch'],ascending=[False,True,True]).iloc[0]
        choice=selected[selected.run.eq(stem)].iloc[0]
        assert choice.selected_epoch==expected.epoch
    frozen=json.loads((out/'selection_frozen.json').read_text())
    for path,digest in frozen['checkpoints'].items():
        assert hashlib.sha256((out/path).read_bytes()).hexdigest()==digest
    print('PASS: 21 seed/evaluation runs; exact frozen test coverage; independent metrics; selection separation; checkpoints; 105 protected files unchanged.')

if __name__=='__main__':main()
