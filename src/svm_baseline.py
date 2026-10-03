"""Training-only StandardScaler + SVC; validation selection; frozen test evaluation."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / '.python_deps'))
import json
import hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sklearn
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.metrics import precision_recall_curve, auc, average_precision_score
import joblib
from extract_features import FEATURES
from dsp_baseline import metrics, require, validate_manifests

OUT = ROOT / 'results/svm'
LIMITATION = ('All 3558 experimental no-flaw images are exact duplicates. '
    'Negative examples across partitions are not independent. Specificity on this '
    'repeated background does NOT demonstrate specificity generalisation.')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit_features(data):
    records = []
    for name in FEATURES:
        values = data[name]
        std = float((values-values.iloc[0]).std(ddof=0))
        dominant = float(values.value_counts(normalize=True).iloc[0])
        constant = values.nunique() == 1
        records.append(dict(feature=name, minimum=values.min(), maximum=values.max(),
            mean=values.mean(), std=std, unique_values=values.nunique(),
            constant=constant, near_constant=bool(constant or dominant >= .99 or std <= 1e-6*max(abs(values.mean()),1)),
            dominant_fraction=dominant, retained=True))
    pd.DataFrame(records).to_csv(OUT/'feature_audit.csv',index=False)
    corr = data[FEATURES].corr()
    corr.to_csv(OUT/'feature_correlation.csv')
    redundant = [dict(feature_a=a,feature_b=b,pearson_r=corr.loc[a,b]) for i,a in enumerate(FEATURES)
                 for b in FEATURES[i+1:] if abs(corr.loc[a,b]) >= .95]
    pd.DataFrame(redundant,columns=['feature_a','feature_b','pearson_r']).to_csv(OUT/'redundant_features.csv',index=False)
    fig, ax=plt.subplots(figsize=(10,9),layout='constrained')
    im=ax.imshow(corr,vmin=-1,vmax=1,cmap='RdBu_r')
    ax.set(xticks=range(12),yticks=range(12),xticklabels=FEATURES,yticklabels=FEATURES,
           title='Feature correlations (descriptive only; all 12 retained)')
    plt.setp(ax.get_xticklabels(),rotation=60,ha='right')
    fig.colorbar(im,ax=ax)
    fig.savefig(OUT/'feature_correlation.png',dpi=150);plt.close(fig)


def grid():
    # Fixed tie order: linear before RBF; ascending C; gamma order as requested.
    return ([dict(kernel='linear',C=c,gamma='scale') for c in [.01,.1,1,10,100]] +
            [dict(kernel='rbf',C=c,gamma=g) for c in [.1,1,10,100] for g in ['scale',.001,.01,.1,1]])


def fit_select(train, validation, candidates=None):
    require(train.partition.eq('train').all(), 'Fit accepts train only')
    require(validation.partition.eq('validation').all(),'Selection accepts validation only')
    require(not set(train.sample_id)&set(validation.sample_id),'Train/validation overlap')
    best, best_objective, best_linear = None, -1, None
    trials=[]
    p=int(validation.has_flaw.sum());n=len(validation)-p
    for index, parameters in enumerate(candidates or grid()):
        model=Pipeline([('scaler',StandardScaler()),('svc',SVC(**parameters,cache_size=256))])
        model.fit(train[FEATURES],train.has_flaw)
        scaler=model.named_steps['scaler']
        require(int(scaler.n_samples_seen_)==len(train),'Scaler used nontraining rows')
        np.testing.assert_allclose(scaler.mean_, train[FEATURES].mean().to_numpy(),rtol=1e-12,atol=1e-9)
        np.testing.assert_allclose(scaler.var_, train[FEATURES].var(ddof=0).to_numpy(),rtol=1e-10,atol=1e-9)
        predicted=model.predict(validation[FEATURES])
        tp=int(((predicted==1)&validation.has_flaw.eq(1)).sum())
        tn=int(((predicted==0)&validation.has_flaw.eq(0)).sum())
        objective=tp*n+tn*p
        trial=dict(candidate=index,**parameters,validation_balanced_accuracy=objective/(2*p*n))
        trials.append(trial)
        if parameters['kernel']=='linear' and (best_linear is None or objective>best_linear[0]):
            best_linear=(objective,model,trial)
        if objective>best_objective:
            best_objective=objective;best=(model,trial)
    return best[0],best[1],trials,best_linear


def confusion(ax, row, title):
    matrix=np.array([[row['TN'],row['FP']],[row['FN'],row['TP']]])
    ax.imshow(matrix,cmap='Blues')
    for (i,j),number in np.ndenumerate(matrix):
        ax.text(j,i,str(number),ha='center',va='center',color='white' if number>matrix.max()/2 else 'black')
    ax.set(title=title,xticks=[0,1],yticks=[0,1],xticklabels=['No flaw','Flaw'],
           yticklabels=['No flaw','Flaw'],xlabel='Predicted',ylabel='True')


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    protected=list((ROOT/'results/dsp_baseline').rglob('*'))+list((ROOT/'data/splits').glob('*'))+[ROOT/'src/dsp_baseline.py']
    protected=[p for p in protected if p.is_file()]
    before={str(p.relative_to(ROOT)):sha(p) for p in protected}
    ordinary=pd.read_csv(ROOT/'data/splits/ordinary_split.csv')
    holdout=pd.read_csv(ROOT/'data/splits/source_holdout_folds.csv')
    validate_manifests(ordinary,holdout)
    features=pd.read_csv(ROOT/'data/features_experimental.csv')
    require(len(features)==7000 and features.sample_id.is_unique,'Feature coverage mismatch')
    require(np.isfinite(features[FEATURES]).all().all(),'Nonfinite features')
    joined=ordinary.merge(features.drop(columns='batch'),on='sample_id',validate='one_to_one')
    require(joined.loc[joined.has_flaw.eq(0),'raw_sha256'].nunique()==1,'Expected duplicated negatives')
    audit_features(joined) # descriptive only; cannot alter feature list or candidate grid
    schemes=[('ordinary',0,ordinary)]+[('source_holdout',int(f),m) for f,m in holdout.groupby('fold')]
    fits=[]; selections=[]; trials=[]; provenance=[]; coefficients=[]
    for scheme,fold,manifest in schemes:
        frame=manifest.merge(features.drop(columns='batch'),on='sample_id',validate='one_to_one')
        train=frame.loc[frame.partition.eq('train')];val=frame.loc[frame.partition.eq('validation')]
        test_ids=set(frame.loc[frame.partition.eq('test'),'sample_id'])
        require(not (set(train.sample_id)|set(val.sample_id))&test_ids,'Test leaked into development')
        if scheme=='source_holdout':
            require(not train.source_label.eq(frame.test_source.iloc[0]).any(),'Held-out source in training')
        model,choice,candidates,linear=fit_select(train,val)
        selection=dict(scheme=scheme,fold=fold,**choice)
        selections.append(selection);trials.extend(dict(scheme=scheme,fold=fold,**t) for t in candidates)
        for role,subset in [('scaler_and_svc_fit',train),('model_selection',val)]:
            provenance.extend(dict(scheme=scheme,fold=fold,sample_id=i,role=role) for i in subset.sample_id)
        for name,weight in zip(FEATURES,linear[1].named_steps['svc'].coef_[0]):
            coefficients.append(dict(scheme=scheme,fold=fold,feature=name,standardized_coefficient=weight,
                absolute_coefficient=abs(weight),linear_C=linear[2]['C'],selected_model_kernel=choice['kernel']))
        joblib.dump(model,OUT/f'{scheme}_{fold}_pipeline.joblib')
        fits.append((selection,model,frame))
        print(f'{scheme} {fold}: frozen {choice}',flush=True)
    pd.DataFrame(selections).to_csv(OUT/'selected_parameters.csv',index=False)
    pd.DataFrame(trials).to_csv(OUT/'validation_grid.csv',index=False)
    pd.DataFrame(provenance).to_csv(OUT/'fitting_and_selection_ids.csv',index=False)
    pd.DataFrame(coefficients).to_csv(OUT/'linear_feature_coefficients.csv',index=False)
    print('All seven selections frozen; test evaluation begins.',flush=True)
    summaries=[];predictions=[];per_source=[]
    for selection,model,frame in fits:
        score=model.decision_function(frame[FEATURES]);pred=model.predict(frame[FEATURES])
        output=frame[['sample_id','batch','has_flaw','source_label','flaw_size','partition']].copy()
        output['decision_score']=score;output['predicted_label']=pred
        output['scheme']=selection['scheme'];output['fold']=selection['fold']
        predictions.append(output)
        test=output.loc[output.partition.eq('test')]
        require(test.sample_id.is_unique and set(test.sample_id)==set(frame.loc[frame.partition.eq('test'),'sample_id']), 'Test prediction coverage')
        met=metrics(test.has_flaw,test.decision_score,test.predicted_label)
        precision,recall,_=precision_recall_curve(test.has_flaw,test.decision_score)
        held=frame.test_source.iloc[0] if selection['scheme']=='source_holdout' else 'mixed'
        sizes=';'.join(map(str,sorted(test.loc[test.has_flaw.eq(1),'flaw_size'].unique())))
        summaries.append(dict(**selection,test_source=held,flaw_size=sizes,n_test=len(test),**met,
            PR_AUC=auc(recall,precision),average_precision=average_precision_score(test.has_flaw,test.decision_score)))
        for source,g in test.loc[test.has_flaw.eq(1)].groupby('source_label'):
            per_source.append(dict(scheme=selection['scheme'],fold=selection['fold'],source_flaw=source,
                flaw_size=g.flaw_size.iloc[0],n_flaw=len(g),svm_recall=g.predicted_label.mean()))
    summary=pd.DataFrame(summaries);summary.to_csv(OUT/'svm_summary.csv',index=False)
    pd.concat(predictions,ignore_index=True).to_csv(OUT/'per_sample_results.csv',index=False)
    dsp=pd.read_csv(ROOT/'results/dsp_baseline/dsp_summary.csv')
    comparison=summary[['scheme','fold','test_source','flaw_size','recall','balanced_accuracy','ROC_AUC']].merge(
        dsp[['scheme','fold','recall','balanced_accuracy','ROC_AUC']],on=['scheme','fold'],suffixes=('_svm','_dsp'),validate='one_to_one')
    comparison.to_csv(OUT/'dsp_vs_svm.csv',index=False)
    source=pd.DataFrame(per_source).merge(pd.read_csv(ROOT/'results/dsp_baseline/per_source_recall.csv')[
        ['scheme','fold','source_flaw','recall']],on=['scheme','fold','source_flaw']).rename(columns={'recall':'dsp_recall'})
    source['recall_difference']=source.svm_recall-source.dsp_recall
    source.to_csv(OUT/'per_source_comparison.csv',index=False)
    held=comparison.loc[comparison.scheme.eq('source_holdout')]
    for metric in ['balanced_accuracy','recall']:
        fig,ax=plt.subplots(figsize=(10,4),layout='constrained');x=np.arange(6)
        ax.bar(x-.18,held[f'{metric}_dsp'],width=.36,label='DSP V1')
        ax.bar(x+.18,held[f'{metric}_svm'],width=.36,label='SVM')
        ax.set(xticks=x,xticklabels=[f'{r.test_source}\n{r.flaw_size} mm' for r in held.itertuples()],
               ylim=(0,1.1),ylabel=metric,title='Held-out-source comparison; duplicated no-flaw background')
        ax.legend();fig.savefig(OUT/f'dsp_vs_svm_{metric}.png',dpi=150);plt.close(fig)
    fig,ax=plt.subplots(figsize=(5,4),layout='constrained');confusion(ax,summary.iloc[0],'Ordinary SVM test')
    fig.savefig(OUT/'ordinary_confusion_matrix.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(12,7),layout='constrained')
    for ax,(_,row) in zip(axes.flat,summary.iloc[1:].iterrows()):confusion(ax,row,row.test_source)
    fig.suptitle('Source-holdout SVM tests; no-flaw images are duplicates')
    fig.savefig(OUT/'source_holdout_confusion_matrices.png',dpi=150);plt.close(fig)
    # Illustrative features fixed before results; no test-informed feature selection.
    names=['max_envelope','rf_rms','crest_factor','spatial_persistence']
    groups=['no_flaw','P41_01','P41_02','P41_03','P41_04','P41_05','P41_06_notch']
    joined['group']=joined.source_label.fillna('no_flaw')
    development=joined.loc[joined.partition.isin(['train','validation'])]
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    for ax,name in zip(axes.flat,names):
        ax.boxplot([development.loc[development.group.eq(g),name] for g in groups],tick_labels=groups)
        ax.set_title(name);ax.tick_params(axis='x',labelrotation=35)
    fig.suptitle('Ordinary training/validation feature distributions; no-flaw values coincide')
    fig.savefig(OUT/'feature_distributions.png',dpi=150);plt.close(fig)
    # PCA is diagnostic only; both PCA and its scaler fit ordinary TRAINING only.
    training=joined.loc[joined.partition.eq('train')]
    pca=Pipeline([('scaler',StandardScaler()),('pca',PCA(n_components=2,svd_solver='full'))])
    pca.fit(training[FEATURES]);coords=pca.transform(joined[FEATURES])
    fig,ax=plt.subplots(figsize=(8,6),layout='constrained')
    for group in groups:
        mask=joined.group.eq(group);ax.scatter(coords[mask,0],coords[mask,1],s=12,alpha=.5,label=group)
    ax.set(title='Diagnostic PCA: fitted on ordinary train, all samples displayed\n3558 identical no-flaw rows overlap',
           xlabel='PC1',ylabel='PC2');ax.legend(fontsize=8)
    fig.savefig(OUT/'pca_diagnostic.png',dpi=150);plt.close(fig)
    (OUT/'pca_details.json').write_text(json.dumps(dict(fit_partition='ordinary train',n_fit=len(training),
        explained_variance_ratio=pca.named_steps['pca'].explained_variance_ratio_.tolist(),classifier_uses_pca=False),indent=2))
    after={str(p.relative_to(ROOT)):sha(p) for p in protected}
    require(before==after,'Protected baseline/split files modified')
    (OUT/'checks.json').write_text(json.dumps(dict(protected_sha256=before,all_unchanged=True,
        scaler_fit_train_only=True,selection_validation_only=True,refit_train_validation=False,
        n_features=12,features=FEATURES,removed_features=[],sklearn=sklearn.__version__,
        limitation=LIMITATION),indent=2))
    print(summary.to_string(index=False));print(LIMITATION)


if __name__=='__main__':main()
