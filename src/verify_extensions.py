"""Independent checks of saved extension outputs; no training or tuning."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.io import loadmat
from sklearn.metrics import confusion_matrix,balanced_accuracy_score,roc_auc_score
from complete_extensions import ROOT,protected,sha

def check_metrics(row,y,score,pred):
    tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    assert (tn,fp,fn,tp)==(row.TN,row.FP,row.FN,row.TP)
    assert abs(balanced_accuracy_score(y,pred)-row.balanced_accuracy)<1e-12
    if score is not None:assert abs(roc_auc_score(y,score)-row.ROC_AUC)<1e-12

def main():
    protected()
    previous=json.loads((ROOT/'docs/methodology/pre_extension_preservation.json').read_text())
    for name,digest in previous.items():
        assert sha(ROOT/name)==digest,name
    print(f'PASS: all {len(previous)} pre-extension scientific files unchanged, including original CNN outputs.')
    # Source-transfer tables can be checked without the raw simulated data or CNN cache.
    out=ROOT/'results/simulated_to_real'
    if (out/'summary.csv').exists():
        summary=pd.read_csv(out/'summary.csv');p=pd.read_csv(out/'predictions.csv')
        split=pd.read_csv(out/'simulated_split.csv');real=pd.read_csv(ROOT/'data/splits/ordinary_split.csv')
        for row in summary.itertuples():
            subset=p[p.domain.eq(row.domain)&p.model.eq(row.model)&p.seed.eq(row.seed)]
            expected=real if row.domain=='experimental_target' else split[split.partition.eq('test')]
            assert subset.sample_id.is_unique and set(subset.sample_id)==set(expected.sample_id)
            np.testing.assert_array_equal(subset.set_index('sample_id').loc[expected.sample_id,'has_flaw'],expected.has_flaw)
            check_metrics(row,subset.has_flaw,subset.score,subset.prediction)
        disclosure=json.loads((out/'shared_background_disclosure.json').read_text())
        assert disclosure['positive_source_target_overlap']==0 and disclosure['target_shared_rows']==3558
        print('PASS: simulated transfer coverage and metrics; shared-background limitation disclosed.')
    # Noise logits and hard decisions are compactly saved for all original rows.
    out=ROOT/'results/noise_robustness';data=np.load(out/'predictions.npz')
    dsp_scores=np.load(out/'dsp_score_evidence.npz')
    np.testing.assert_array_equal(dsp_scores['sample_ids'],data['sample_ids'])
    np.testing.assert_array_equal(dsp_scores['noise_seeds'],data['noise_seeds'])
    np.testing.assert_array_equal(dsp_scores['snr_db'],data['snr_db'])
    rows=pd.read_csv(out/'summary.csv');hold=pd.read_csv(ROOT/'data/splits/source_holdout_folds.csv')
    index=pd.Series(np.arange(len(data['sample_ids'])),index=data['sample_ids'])
    for row in rows.itertuples():
        frame=hold[hold.fold.eq(row.fold)&hold.partition.eq('test')];ix=index.loc[frame.sample_id].to_numpy()
        r=list(data['noise_seeds']).index(row.noise_seed);s=list(data['snr_db']).index(row.snr_db)
        if row.model=='CNN':
            k=list(data['model_seeds']).index(row.model_seed);score=data['cnn_logits'][r,s,k,ix];pred=score>=0
        else:score=dsp_scores['scores'][r,s,ix];pred=data['dsp_predictions'][r,s,ix]
        check_metrics(row,frame.has_flaw,score,pred)
    assert len(rows)==360
    print('PASS: 360 noise method/seed/fold/SNR rows, frozen test membership and independent metric checks.')
    # Verify reported localization directly from saved reconstruction grids.
    out=ROOT/'results/fmc_tfm'
    for result_name,image_name in [('localisation.json','tfm_image.npz'),('calibrated_localisation.json','calibrated_tfm_image.npz')]:
        result=json.loads((out/result_name).read_text());image=np.load(out/image_name)
        xx,zz=np.meshgrid(image['x_m']*1000,image['z_m']*1000)
        mask=(xx>=29)&(xx<=41)&(zz>=17)&(zz<=21.5)
        k=np.unravel_index(np.where(mask,image['amplitude'],-np.inf).argmax(),xx.shape)
        estimate=np.array([xx[k],zz[k]])
        np.testing.assert_allclose(estimate,result['estimate_x_z_mm'],atol=1e-12)
        assert abs(np.linalg.norm(estimate-[35,20])-result['euclidean_error_mm'])<1e-12
    print('PASS: initial and backwall-calibrated TFM ROI peaks and nominal-point errors; 105 baseline files unchanged.')

if __name__=='__main__':main()
