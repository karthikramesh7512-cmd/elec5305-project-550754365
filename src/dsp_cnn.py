"""Additive final-stage DSP-envelope CNN experiment; frozen baseline outputs are read-only.

Run with --dataset-dir <folder containing batch_013.xz>. Cache is local, never published.
Training all checkpoints precedes evaluation of any test rows. No test-driven reruns.
"""
from pathlib import Path
import argparse, copy, hashlib, json, lzma, os, random, time
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import pandas as pd
import scipy
from scipy.signal import hilbert
import torch
from torch import nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from dsp_baseline import metrics, validate_manifests

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/dsp_cnn'
CACHE = ROOT / '.cnn_cache'
CONFIG = dict(seeds=[42,43,44], max_epochs=20, patience=5, batch_size=64,
    learning_rate=0.001, weight_decay=0.0001, threshold=0.5,
    roi=[1100,3100], input_shape=[60,125], pooling=[8,16],
    preprocessing='per-A-scan DC removal; Hilbert magnitude; 8x16 block maximum; log1p',
    normalization='one scalar mean/std fitted on training maps per fold only',
    selection='maximum validation balanced accuracy, then minimum validation BCE; earliest exact tie',
    architecture='Conv3x3(1,8)-ReLU-MaxPool2; Conv3x3(8,16)-ReLU-MaxPool2; Conv3x3(16,32)-ReLU; global mean+max; Linear(64,1)',
    augmentation='none', loss='unweighted BCEWithLogitsLoss', optimizer='Adam',
    evaluation='six frozen source-holdout folds primary; ordinary split reference; three seeds')

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1048576),b''): h.update(block)
    return h.hexdigest()

def write_json(path, data):
    path.write_text(json.dumps(data,indent=2),encoding='utf-8')

def envelope_map(image):
    if image.shape != (480,7168): raise ValueError('Expected raw B-scan shape (480,7168)')
    roi=image[:,1100:3100].astype(np.float64)
    roi-=roi.mean(axis=1,keepdims=True)
    envelope=np.abs(hilbert(roi,axis=1))
    pooled=envelope.reshape(60,8,125,16).max(axis=(1,3))
    return np.log1p(pooled).astype(np.float32)

class CompactCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features=nn.Sequential(nn.Conv2d(1,8,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),
            nn.Conv2d(8,16,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),nn.Conv2d(16,32,3,padding=1),nn.ReLU())
        self.classifier=nn.Linear(64,1)
    def forward(self,x):
        x=self.features(x)
        return self.classifier(torch.cat([x.mean((2,3)),x.amax((2,3))],dim=1)).squeeze(1)

def build_cache(base, dataset):
    CACHE.mkdir(exist_ok=True)
    feature_path=ROOT/'data/features_experimental.csv'
    provenance=dict(ordinary_sha256=sha(ROOT/'data/splits/ordinary_split.csv'),
        features_sha256=sha(feature_path),code_sha256=sha(Path(__file__)),config=CONFIG,
        raw_files={p.name:sha(p) for p in sorted(dataset.glob('batch_0*.xz')) if p.stem in set(base.batch)})
    cache_path=CACHE/'envelope_maps.npy'
    meta_path=CACHE/'provenance.json'
    if meta_path.exists() and json.loads(meta_path.read_text())==provenance and cache_path.exists():
        maps=np.load(cache_path)
        if maps.shape==(7000,60,125) and np.isfinite(maps).all(): return maps,provenance
    maps=np.empty((7000,60,125),dtype=np.float32)
    audit=pd.read_csv(feature_path).set_index('sample_id')
    cached_by_hash={}
    for batch in sorted(base.batch.unique()):
        with lzma.open(dataset/f'{batch}.xz','rb') as stream:
            for row in base.loc[base.batch.eq(batch)].sort_values('metadata_row').itertuples():
                payload=stream.read(480*7168*2)
                if len(payload)!=480*7168*2: raise ValueError('Truncated raw image')
                digest=hashlib.sha256(payload).hexdigest()
                if digest!=audit.loc[row.sample_id,'raw_sha256']: raise ValueError('Raw sample mismatch '+row.sample_id)
                if digest not in cached_by_hash:
                    cached_by_hash[digest]=envelope_map(np.frombuffer(payload,dtype='<i2').reshape(480,7168))
                maps[row.Index]=cached_by_hash[digest]
                if row.metadata_row%250==0: print(f'Preprocess {batch}: {row.metadata_row}/1000',flush=True)
            if stream.read(1): raise ValueError('Unexpected trailing raw data')
    if not np.isfinite(maps).all(): raise ValueError('Nonfinite CNN inputs')
    np.save(cache_path,maps)
    write_json(meta_path,provenance)
    return maps,provenance

def score(model,x,device):
    model.eval()
    with torch.no_grad(): return torch.cat([model(b.to(device)).cpu() for b in x.split(128)]).numpy()

def fit_fold(frame,maps,base_index,seed,device,stem):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
    groups={part:frame.loc[frame.partition.eq(part)] for part in ['train','validation','test']}
    ids={k:set(v.sample_id) for k,v in groups.items()}
    assert not ids['train']&ids['validation'] and not ids['test']&(ids['train']|ids['validation'])
    train_idx=base_index.loc[groups['train'].sample_id].to_numpy()
    val_idx=base_index.loc[groups['validation'].sample_id].to_numpy()
    # Test maps and labels are not loaded during fitting or checkpoint selection.
    mean=float(maps[train_idx].mean(dtype=np.float64));std=float(maps[train_idx].std(dtype=np.float64))
    if std<=0: raise ValueError('Degenerate training normalization')
    xt=torch.from_numpy((maps[train_idx,None]-mean)/std)
    xv=torch.from_numpy((maps[val_idx,None]-mean)/std)
    yt=torch.tensor(groups['train'].has_flaw.to_numpy(),dtype=torch.float32)
    yv=groups['validation'].has_flaw.to_numpy()
    model=CompactCNN().to(device);optimizer=torch.optim.Adam(model.parameters(),lr=CONFIG['learning_rate'],weight_decay=CONFIG['weight_decay'])
    loss_fn=nn.BCEWithLogitsLoss();best=None;history=[];stale=0;start=time.perf_counter()
    for epoch in range(1,CONFIG['max_epochs']+1):
        model.train();order=torch.randperm(len(xt));total=0.
        for batch in order.split(CONFIG['batch_size']):
            optimizer.zero_grad(set_to_none=True)
            loss=loss_fn(model(xt[batch].to(device)),yt[batch].to(device))
            loss.backward();optimizer.step();total+=loss.item()*len(batch)
        logits=score(model,xv,device)
        val_loss=float(loss_fn(torch.tensor(logits),torch.tensor(yv,dtype=torch.float32)))
        predicted=logits>=0
        ba=float(((predicted[yv==1]).mean()+(~predicted[yv==0]).mean())/2)
        history.append(dict(epoch=epoch,training_loss=total/len(xt),validation_loss=val_loss,validation_balanced_accuracy=ba))
        key=(ba,-val_loss)
        if best is None or key>best[0]:best=(key,copy.deepcopy(model.state_dict()),epoch);stale=0
        else:stale+=1
        print(f'{stem}: epoch {epoch}, val BA={ba:.4f}, loss={val_loss:.5f}',flush=True)
        if stale>=CONFIG['patience']:break
    torch.save(dict(state_dict=best[1],mean=mean,std=std,config=CONFIG,seed=seed,selected_epoch=best[2]),OUT/f'{stem}.pt')
    pd.DataFrame(history).to_csv(OUT/f'{stem}_history.csv',index=False)
    pd.concat([groups[k].assign(role=k) for k in ['train','validation']]).to_csv(OUT/f'{stem}_selection_ids.csv',index=False)
    return dict(run=stem,seed=seed,selected_epoch=best[2],validation_balanced_accuracy=best[0][0],
        validation_loss=-best[0][1],training_seconds=time.perf_counter()-start,training_mean=mean,training_std=std,
        parameter_count=sum(p.numel() for p in model.parameters()))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--dataset-dir',type=Path,required=True);args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'cnn_summary.csv').exists(): raise RuntimeError('Completed experiment exists; refusing a test-driven rerun')
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ordinary=pd.read_csv(ROOT/'data/splits/ordinary_split.csv');holdout=pd.read_csv(ROOT/'data/splits/source_holdout_folds.csv')
    validate_manifests(ordinary,holdout)
    protected=json.loads((ROOT/'docs/methodology/packaging_preservation.json').read_text())['protected_sha256']
    assert all(sha(ROOT/p)==h for p,h in protected.items()),'Baseline preservation failure before experiment'
    write_json(OUT/'experiment_config.json',dict(**CONFIG,device=str(device),torch_version=torch.__version__,
        numpy_version=np.__version__,scipy_version=scipy.__version__,created_utc=pd.Timestamp.now(tz='UTC').isoformat(),
        source_sha256=sha(Path(__file__)),protected_sha256=protected))
    print('Device:',device,flush=True)
    maps,provenance=build_cache(ordinary,args.dataset_dir)
    write_json(OUT/'input_provenance.json',dict(**provenance,cache_sha256=sha(CACHE/'envelope_maps.npy')))
    base_index=pd.Series(np.arange(len(ordinary)),index=ordinary.sample_id)
    schemes=[('ordinary',0,ordinary)]+[('source_holdout',int(f),frame) for f,frame in holdout.groupby('fold')]
    selections=[]
    for scheme,fold,frame in schemes:
        for seed in CONFIG['seeds']:
            stem=f'{scheme}_{fold}_seed_{seed}'
            selected=fit_fold(frame,maps,base_index,seed,device,stem)
            selections.append(dict(scheme=scheme,fold=fold,**selected))
            pd.DataFrame(selections).to_csv(OUT/'checkpoint_selection.csv',index=False)
    write_json(OUT/'selection_frozen.json',dict(checkpoints={p.name:sha(p) for p in OUT.glob('*.pt')},test_evaluation_started=False))
    # All checkpoint selection is finished. First and only test evaluation follows.
    summaries=[];predictions=[]
    for scheme,fold,frame in schemes:
        test=frame.loc[frame.partition.eq('test')].copy();idx=base_index.loc[test.sample_id].to_numpy()
        for seed in CONFIG['seeds']:
            stem=f'{scheme}_{fold}_seed_{seed}'
            ckpt=torch.load(OUT/f'{stem}.pt',map_location=device,weights_only=False)
            model=CompactCNN().to(device);model.load_state_dict(ckpt['state_dict'])
            x=torch.from_numpy((maps[idx,None]-ckpt['mean'])/ckpt['std'])
            logits=score(model,x,device);prob=torch.sigmoid(torch.tensor(logits)).numpy();pred=logits>=0
            source='mixed' if scheme=='ordinary' else str(frame.test_source.iloc[0])
            summaries.append(dict(scheme=scheme,fold=fold,seed=seed,test_source=source,n_test=len(test),**metrics(test.has_flaw.to_numpy(),logits,pred)))
            predictions.append(test[['sample_id','has_flaw']].assign(scheme=scheme,fold=fold,seed=seed,logit=logits,probability=prob,prediction=pred.astype(int)))
    summary=pd.DataFrame(summaries);summary.to_csv(OUT/'cnn_summary.csv',index=False)
    pd.concat(predictions).to_csv(OUT/'per_sample_results.csv',index=False)
    for metric in ['recall','balanced_accuracy']:
        means=summary.loc[summary.scheme.eq('source_holdout')].groupby(['fold','test_source'])[metric].agg(['mean','std','min','max']).reset_index()
        baseline=pd.read_csv(ROOT/'results/dsp_baseline/dsp_summary.csv').query("scheme == 'source_holdout'")
        svm=pd.read_csv(ROOT/'results/svm/svm_summary.csv').query("scheme == 'source_holdout'")
        means=means.merge(baseline[['fold',metric]].rename(columns={metric:'DSP'}),on='fold').merge(svm[['fold',metric]].rename(columns={metric:'SVM'}),on='fold')
        means.to_csv(OUT/f'comparison_{metric}.csv',index=False)
        fig,ax=plt.subplots(figsize=(9,4.7),layout='constrained');x=np.arange(6)
        for off,col,label,color in [(-.25,'DSP','DSP','#657783'),(0,'SVM','DSP + SVM (reference)','#b3bec5'),(.25,'mean','DSP + CNN (3 seeds)','#22678a')]:
            ax.bar(x+off,100*means[col],.24,label=label,color=color)
        ax.errorbar(x+.25,100*means['mean'],yerr=100*means['std'],fmt='none',color='black',capsize=3)
        ax.set(xticks=x,xticklabels=means.test_source,ylim=(0,112),ylabel=metric.replace('_',' ').title()+' (%)',title='Frozen source-holdout evaluation')
        ax.tick_params(axis='x',labelsize=9);ax.legend(fontsize=9,loc='lower right');ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
        fig.savefig(OUT/f'comparison_{metric}.png',dpi=300);fig.savefig(OUT/f'comparison_{metric}.svg');plt.close(fig)
    assert all(sha(ROOT/p)==h for p,h in protected.items()),'Baseline changed during experiment'
    write_json(OUT/'completion_checks.json',dict(protected_files_unchanged=len(protected),all_test_ids_match_frozen_manifests=True,
        models=21,seeds=CONFIG['seeds'],test_used_for_selection=False,
        limitation='All 3558 no-flaw rows repeat one raw image; specificity is not independent-background generalisation. Related augmented positives also limit independence. Seeds measure initialization variation, not independent datasets.'))
    print(summary.to_string(index=False),flush=True)

if __name__=='__main__':main()
