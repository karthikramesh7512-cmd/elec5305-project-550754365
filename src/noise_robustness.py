"""Frozen-model AWGN stress test on the existing six source-holdout test sets.

Noise is applied before envelope extraction, never to features or model outputs.
Exact duplicate raw images share each realization; no independence claim is added.
"""
from pathlib import Path
import argparse, hashlib, json, lzma
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import dsp_cnn as cnn
from dsp_baseline import metrics
from complete_extensions import PLAN, setup, protected, sha, dump

ROOT=Path(__file__).resolve().parents[1];CACHE=ROOT/'.extension_cache';OUT=ROOT/'results/noise_robustness'
SNR=[20,15,10,5,0];REPS=[101,102,103];SEEDS=[42,43,44];BATCH=8

def gpu_envelope(rf):
    n=rf.shape[-1];h=torch.zeros(n,dtype=rf.dtype,device=rf.device);h[0]=h[n//2]=1;h[1:n//2]=2
    return torch.fft.ifft(torch.fft.fft(rf,dim=-1)*h,dim=-1).abs()

def add_noise(rf,snr,noise):
    noise=noise-noise.mean(dim=-1,keepdim=True)
    rms=rf.square().mean((1,2),keepdim=True).sqrt()
    n_rms=noise.square().mean((1,2),keepdim=True).sqrt()
    return rf+noise/n_rms*rms*(10**(-snr/20))

def prepare(dataset,base,features):
    hashes=features.set_index('sample_id').loc[base.sample_id,'raw_sha256'].to_numpy()
    unique=list(dict.fromkeys(hashes));lookup={h:i for i,h in enumerate(unique)}
    positions=np.array([lookup[h] for h in hashes])
    provenance=dict(protocol=PLAN,raw_hashes=unique,batch_size=BATCH,code_sha256=sha(Path(__file__)),
        input_features_sha256=sha(ROOT/'data/features_experimental.csv'),dtype='float64 FFT and noise; float32 stored maps',
        duplicate_policy='Identical raw images share the realization within each replicate and SNR',rng='CUDA generator seed = replicate*100000 + unique-batch start index; noise reused across SNR levels')
    meta_path=CACHE/'noise_provenance.json'
    shape=(3,5,len(unique),60,125)
    if meta_path.exists() and json.loads(meta_path.read_text())==provenance:
        return np.load(CACHE/'noisy_maps.npy',mmap_mode='r'),np.load(CACHE/'noisy_dsp_scores.npy'),positions
    bank=np.lib.format.open_memmap(CACHE/'noisy_maps.npy',mode='w+',dtype=np.float32,shape=shape)
    scores=np.empty((3,5,len(unique)))
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seen=set();pending=[];largest_snr_error=0.
    def flush():
        nonlocal largest_snr_error
        if not pending:return
        indices=[x[0] for x in pending]
        rf=torch.from_numpy(np.stack([x[1] for x in pending])).to(device)
        rf=rf-rf.mean(-1,keepdim=True)
        # Clean FFT implementation must agree with the saved CPU representation.
        env=gpu_envelope(rf)
        clean_maps=torch.log1p(env.reshape(-1,60,8,125,16).amax((2,4))).float().cpu().numpy()
        reference=np.load(ROOT/'.cnn_cache/envelope_maps.npy',mmap_mode='r')
        for k,(uid,_,sample_position) in enumerate(pending):np.testing.assert_allclose(clean_maps[k],reference[sample_position],atol=2e-6,rtol=2e-6)
        for ri,rep in enumerate(REPS):
            generator=torch.Generator(device=device);generator.manual_seed(rep*100000+indices[0])
            noise=torch.randn(rf.shape,dtype=torch.float64,device=device,generator=generator)
            for si,snr in enumerate(SNR):
                noisy=add_noise(rf,snr,noise)
                realized=10*torch.log10(rf.square().mean((1,2))/(noisy-rf).square().mean((1,2)))
                largest_snr_error=max(largest_snr_error,float((realized-snr).abs().max()))
                envelope=gpu_envelope(noisy)
                bank[ri,si,indices]=torch.log1p(envelope.reshape(-1,60,8,125,16).amax((2,4))).float().cpu().numpy()
                scores[ri,si,indices]=envelope.amax((1,2)).cpu().numpy()
        pending.clear()
    audit=features.set_index('sample_id')
    for batch in sorted(base.batch.unique()):
        with lzma.open(dataset/f'{batch}.xz','rb') as f:
            for row in base[base.batch.eq(batch)].sort_values('metadata_row').itertuples():
                payload=f.read(6881280);assert len(payload)==6881280
                digest=hashlib.sha256(payload).hexdigest();assert digest==audit.loc[row.sample_id,'raw_sha256']
                if digest not in seen:
                    seen.add(digest);image=np.frombuffer(payload,dtype='<i2').reshape(480,7168)
                    pending.append((lookup[digest],image[:,1100:3100].astype(float),row.Index))
                    if len(pending)==BATCH:flush()
        print('Noisy preprocessing:',batch,'unique images',len(seen),'/',len(unique),flush=True)
    flush();assert len(seen)==len(unique)
    bank.flush();np.save(CACHE/'noisy_dsp_scores.npy',scores)
    dump(meta_path,provenance)
    dump(OUT/'input_audit.json',dict(provenance=provenance,unique_images=len(unique),rows=len(base),max_absolute_snr_error_db=largest_snr_error))
    return bank,scores,positions

def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset-dir',type=Path,required=True);a=p.parse_args();setup();protected();OUT.mkdir(exist_ok=True)
    if (OUT/'summary.csv').exists():raise RuntimeError('Completed noise experiment exists; refusing overwrite')
    base=pd.read_csv(ROOT/'data/splits/ordinary_split.csv');hold=pd.read_csv(ROOT/'data/splits/source_holdout_folds.csv')
    features=pd.read_csv(ROOT/'data/features_experimental.csv');bank,scores,positions=prepare(a.dataset_dir,base,features)
    index=pd.Series(np.arange(len(base)),index=base.sample_id)
    parameters=pd.read_csv(ROOT/'results/dsp_baseline/selected_parameters.csv').query("scheme=='source_holdout'")
    assert parameters.window.eq(1).all()
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    rows=[];logits=np.empty((3,5,3,len(base)),np.float32);dsp_prediction=np.empty((3,5,len(base)),np.int8)
    frozen={p.name:sha(p) for p in (ROOT/'results/dsp_cnn').glob('source_holdout*.pt')}
    dump(OUT/'frozen_protocol.json',dict(protocol=PLAN,checkpoints=frozen,source_code_sha256=sha(Path(__file__))))
    for fold,frame in hold.groupby('fold'):
        test=frame[frame.partition.eq('test')];ix=index.loc[test.sample_id].to_numpy();uids=positions[ix]
        threshold=float(parameters[parameters.fold.eq(fold)].threshold.iloc[0])
        models=[]
        for seed in SEEDS:
            checkpoint=torch.load(ROOT/f'results/dsp_cnn/source_holdout_{fold}_seed_{seed}.pt',map_location=device,weights_only=False)
            model=cnn.CompactCNN().to(device);model.load_state_dict(checkpoint['state_dict']);models.append((model,checkpoint))
        for ri,rep in enumerate(REPS):
            for si,snr in enumerate(SNR):
                sc=scores[ri,si,uids];pr=sc>=threshold;dsp_prediction[ri,si,ix]=pr.astype(int)
                rows.append(dict(model='DSP',fold=int(fold),source=frame.test_source.iloc[0],noise_seed=rep,model_seed=0,snr_db=snr,n=len(test),**metrics(test.has_flaw,sc,pr)))
                for ki,(seed,(model,checkpoint)) in enumerate(zip(SEEDS,models)):
                    x=torch.from_numpy((np.array(bank[ri,si,uids,None])-checkpoint['mean'])/checkpoint['std'])
                    sc=cnn.score(model,x,device);logits[ri,si,ki,ix]=sc
                    rows.append(dict(model='CNN',fold=int(fold),source=frame.test_source.iloc[0],noise_seed=rep,model_seed=seed,snr_db=snr,n=len(test),**metrics(test.has_flaw,sc,sc>=0)))
        print('Noisy evaluation complete, fold',fold,flush=True)
    pd.DataFrame(rows).to_csv(OUT/'summary.csv',index=False)
    np.savez_compressed(OUT/'predictions.npz',sample_ids=base.sample_id.to_numpy(dtype=str),labels=base.has_flaw.to_numpy(),
        snr_db=SNR,noise_seeds=REPS,model_seeds=SEEDS,cnn_logits=logits,dsp_predictions=dsp_prediction)
    group=pd.DataFrame(rows).groupby(['model','snr_db','noise_seed','model_seed'])[['recall','balanced_accuracy','specificity','ROC_AUC']].mean().reset_index()
    group.to_csv(OUT/'macro_by_replicate.csv',index=False)
    agg=group.groupby(['model','snr_db'])[['recall','balanced_accuracy','specificity','ROC_AUC']].agg(['mean','std']).reset_index()
    agg.to_csv(OUT/'macro_summary.csv',index=False)
    fig,axes=plt.subplots(1,2,figsize=(9,3.7),layout='constrained')
    clean_dsp=pd.read_csv(ROOT/'results/dsp_baseline/dsp_summary.csv').query("scheme=='source_holdout'")
    clean_cnn=pd.read_csv(ROOT/'results/dsp_cnn/cnn_summary.csv').query("scheme=='source_holdout'")
    for ax,metric in zip(axes,['recall','balanced_accuracy']):
        for name,color in [('DSP','#687b85'),('CNN','#216789')]:
            d=group[group.model.eq(name)].groupby('snr_db')[metric].agg(['mean','std']).reindex([0,5,10,15,20])
            ax.errorbar([0,5,10,15,20],100*d['mean'],yerr=100*d['std'],label=name,color=color,marker='o',capsize=3)
            clean=(clean_dsp if name=='DSP' else clean_cnn)[metric].mean()
            ax.axhline(clean*100,color=color,ls='--',lw=.9)
        ax.set(xlabel='Added-noise SNR in centred ROI (dB)',ylabel=metric.replace('_',' ').title()+' (%)',ylim=(0,105));ax.grid(alpha=.2);ax.legend()
    fig.savefig(OUT/'noise_performance.png',dpi=300);fig.savefig(OUT/'noise_performance.svg');plt.close(fig)
    dump(OUT/'completion.json',dict(baseline_files_unchanged=protected(),new_training=False,
        interpretation='Descriptive synthetic AWGN stress test. Duplicate backgrounds share noise. Variation combines 3 noise draws and 3 CNN seeds; not a confidence interval for physical acquisitions. SVM was not included in this noise extension.'))
    print(group.groupby(['model','snr_db'])[['recall','balanced_accuracy','specificity']].mean().to_string(),flush=True)

if __name__=='__main__':main()
