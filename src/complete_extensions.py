"""Additive robustness, simulated-to-real and measured-FMC experiments.

Each stage writes a separate results directory. Existing experiments are read-only.
"""
from pathlib import Path
import argparse, hashlib, json, lzma, os, copy, time
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import numpy as np
import pandas as pd
import torch
from scipy.signal import hilbert, butter, sosfiltfilt, resample_poly
from scipy.io import loadmat
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import dsp_cnn as cnn
from dsp_baseline import metrics, select_parameters, smooth_line
from extract_features import features_from_image, FEATURES

ROOT=Path(__file__).resolve().parents[1]
CACHE=ROOT/'.extension_cache'
SEEDS=[42,43,44]
PLAN=dict(noise_snr_db=[20,15,10,5,0],noise_replicates=[101,102,103],
    noise_definition='AWGN added to DC-centred ROI; noise DC removed per A-scan and scaled to exact ROI RMS SNR. Repeated raw hashes share the same realization. Frozen models and thresholds; no retraining.',
    transfer_train_batches=list(range(201,208)),transfer_validation_batches=[208],transfer_test_batches=[209,210],
    transfer_target='All 7000 experimental rows, no target training or validation; encoded simulated sizes not mapped to millimetres.',
    tfm=dict(sound_speed_m_s=5820,sample_rate_hz=100000000,elements=128,pitch_mm=.7,
        bandpass_mhz=[2,8],decimation=5,grid_step_mm=.25,x_extent_mm=[0,90],z_extent_mm=[1,26],
        nominal_lower_tip_mm=[35,20],localization_search_mm=dict(x=[29,41],z=[17,21.5]),
        time_zero_seconds=0,model='homogeneous longitudinal straight ray; full aperture; linear time interpolation',
        interpretation='Metadata-assisted lower-tip ROI, not blind localisation. Nominal tip is approximate; weld anisotropy and time-zero uncertainty remain.'))

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def dump(path,data):path.write_text(json.dumps(data,indent=2),encoding='utf-8')
def setup():
    CACHE.mkdir(exist_ok=True)
    p=ROOT/'docs/methodology/extension_protocol.json'
    if p.exists():assert json.loads(p.read_text())==PLAN,'Protocol changed'
    else:dump(p,PLAN)
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
def protected():
    p=ROOT/'results/dsp_cnn/experiment_config.json'
    d=json.loads(p.read_text())['protected_sha256']
    for f,h in d.items():assert sha(ROOT/f)==h,f
    return len(d)
def summarize(y,score,pred):return metrics(np.asarray(y),np.asarray(score),np.asarray(pred))

def simulated_transfer(dataset,resume=False):
    out=ROOT/'results/simulated_to_real';out.mkdir(exist_ok=True)
    if (out/'summary.csv').exists():raise RuntimeError('Completed transfer exists; refusing overwrite')
    meta=pd.read_csv(ROOT/'data/metadata.csv');sim=meta[meta.batch_origin.eq('simulated')].reset_index(drop=True)
    assert len(sim)==10000
    sim['partition']=sim.batch.map(lambda b:'train' if int(b[-3:])<=207 else 'validation' if b.endswith('208') else 'test')
    sim.to_csv(out/'simulated_split.csv',index=False)
    if resume:
        maps=np.load(CACHE/'simulated_maps.npy',mmap_mode='r')
        info=pd.read_csv(out/'simulated_features_and_scores.csv')
        return evaluate_simulated_transfer(out,sim,maps,info)
    maps=np.empty((len(sim),60,125),np.float32);records=[];cached={};byte_counts={}
    for batch in sorted(sim.batch.unique()):
        total=0
        with lzma.open(dataset/f'{batch}.lzma','rb') as f:
            for row in sim[sim.batch.eq(batch)].sort_values('metadata_row').itertuples():
                b=f.read(6881280);assert len(b)==6881280
                total+=len(b);digest=hashlib.sha256(b).hexdigest()
                if digest not in cached:
                    image=np.frombuffer(b,dtype='<i2').reshape(480,7168)
                    roi=image[:,1100:3100].astype(float);roi-=roi.mean(1,keepdims=True)
                    env=np.abs(hilbert(roi,axis=1));line=env.max(1)
                    cached[digest]=(np.log1p(env.reshape(60,8,125,16).max((1,3))).astype(np.float32),
                        {f'score_w{w}':float(smooth_line(line,w).max()) for w in [1,3,5,7]},features_from_image(image))
                mp,scores,features=cached[digest];maps[row.Index]=mp
                records.append(dict(sample_id=row.sample_id,raw_sha256=digest,**scores,**features))
            assert not f.read(1)
        byte_counts[batch]=total;print('Simulated preprocessing:',batch,flush=True)
    info=pd.DataFrame(records);info.to_csv(out/'simulated_features_and_scores.csv',index=False)
    np.save(CACHE/'simulated_maps.npy',maps)
    check=sim.merge(info[['sample_id','raw_sha256']],on='sample_id')
    real=pd.read_csv(ROOT/'data/features_experimental.csv')
    overlap=set(info.raw_sha256)&set(real.raw_sha256)
    dump(out/'data_audit.json',dict(row_count=len(sim),stream_bytes=byte_counts,unique_raw_hashes=info.raw_sha256.nunique(),
        negative_unique_hashes=check[check.has_flaw.eq(0)].raw_sha256.nunique(),
        positive_unique_hashes=check[check.has_flaw.eq(1)].raw_sha256.nunique(),source_target_identical_hashes=len(overlap),
        source_target_overlap_hashes=sorted(overlap),decoding='1000 headerless 480x7168 little-endian int16 images per stream',
        coordinates='Simulation coordinate units/encoding differ and are not used. Size-code-to-mm mapping remains unverified.'))
    return evaluate_simulated_transfer(out,sim,maps,info)

def evaluate_simulated_transfer(out,sim,maps,info):
    real=pd.read_csv(ROOT/'data/features_experimental.csv')
    real_meta=pd.read_csv(ROOT/'data/splits/ordinary_split.csv')
    overlap=set(info.raw_sha256)&set(real.raw_sha256)
    source_labels=sim.merge(info[['sample_id','raw_sha256']],on='sample_id')
    target_labels=real_meta.merge(real[['sample_id','raw_sha256']],on='sample_id')
    assert not source_labels[source_labels.raw_sha256.isin(overlap)].has_flaw.any(),'Positive source/target collision'
    assert not target_labels[target_labels.raw_sha256.isin(overlap)].has_flaw.any(),'Positive source/target collision'
    dump(out/'shared_background_disclosure.json',dict(shared_hashes=sorted(overlap),
        source_shared_rows=int(source_labels.raw_sha256.isin(overlap).sum()),target_shared_rows=int(target_labels.raw_sha256.isin(overlap).sum()),
        known_before_model_fitting=True,positive_source_target_overlap=0,
        interpretation='Shared-negative-background transfer. Target specificity is not independent-domain evidence; positive-source transfer is the meaningful part.',
        preprocessing_maps_sha256=sha(CACHE/'simulated_maps.npy'),preprocessing_features_sha256=sha(out/'simulated_features_and_scores.csv')))
    val=sim[sim.partition.eq('validation')][['sample_id','has_flaw','partition']].merge(info,on='sample_id')
    selected,candidates=select_parameters(val)
    dump(out/'dsp_selection.json',dict(selected=selected,candidates=candidates))
    cnn.OUT=out;index=pd.Series(np.arange(len(sim)),index=sim.sample_id)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    selections=[]
    for seed in SEEDS:
        selections.append(cnn.fit_fold(sim,maps,index,seed,device,f'simulated_seed_{seed}'))
    pd.DataFrame(selections).to_csv(out/'cnn_selection.csv',index=False)
    # SVM is trained solely on simulated training rows with the original grid.
    import svm_baseline as svm
    merged=sim.merge(info,on='sample_id');sv,choice,trials,_=svm.fit_select(merged[merged.partition.eq('train')],merged[merged.partition.eq('validation')])
    import joblib
    joblib.dump(sv,out/'simulated_svm.joblib');dump(out/'svm_selection.json',dict(choice=choice,trials=trials))
    dump(out/'selection_frozen.json',dict(config=PLAN,cnn_checkpoints={p.name:sha(p) for p in out.glob('*.pt')},
        svm_sha256=sha(out/'simulated_svm.joblib'),source_code_sha256=sha(Path(__file__))))
    # No target labels/maps have entered fitting or checkpoint selection.
    real_meta=pd.read_csv(ROOT/'data/splits/ordinary_split.csv')
    real_maps=np.load(ROOT/'.cnn_cache/envelope_maps.npy',mmap_mode='r')
    real_scores=pd.read_csv(ROOT/'results/dsp_baseline/candidate_scores.csv').set_index('sample_id')
    score_column=f"score_w{selected['window']}"
    rows=[];predictions=[]
    for domain in ['simulated_test','experimental_target']:
        frame=sim[sim.partition.eq('test')] if domain=='simulated_test' else real_meta
        positions=index.loc[frame.sample_id].to_numpy() if domain=='simulated_test' else np.arange(len(frame))
        bank=maps if domain=='simulated_test' else real_maps
        scores=info.set_index('sample_id').loc[frame.sample_id,score_column].to_numpy() if domain=='simulated_test' else real_scores.loc[frame.sample_id,score_column].to_numpy()
        for model_name,seed,sc,pr in [('DSP',0,scores,scores>=selected['threshold']),
             ('SVM',0,sv.decision_function((info if domain=='simulated_test' else real).set_index('sample_id').loc[frame.sample_id,FEATURES]),sv.predict((info if domain=='simulated_test' else real).set_index('sample_id').loc[frame.sample_id,FEATURES]))]:
            rows.append(dict(domain=domain,model=model_name,seed=seed,n=len(frame),**summarize(frame.has_flaw,sc,pr)))
            predictions.append(frame[['sample_id','has_flaw']].assign(domain=domain,model=model_name,seed=seed,score=sc,prediction=np.asarray(pr).astype(int)))
        for seed in SEEDS:
            ckpt=torch.load(out/f'simulated_seed_{seed}.pt',map_location=device,weights_only=False)
            model=cnn.CompactCNN().to(device);model.load_state_dict(ckpt['state_dict'])
            x=torch.from_numpy((bank[positions,None]-ckpt['mean'])/ckpt['std'])
            logits=cnn.score(model,x,device)
            rows.append(dict(domain=domain,model='CNN',seed=seed,n=len(frame),**summarize(frame.has_flaw,logits,logits>=0)))
            predictions.append(frame[['sample_id','has_flaw']].assign(domain=domain,model='CNN',seed=seed,score=logits,prediction=(logits>=0).astype(int)))
    pd.DataFrame(rows).to_csv(out/'summary.csv',index=False);pd.concat(predictions).to_csv(out/'predictions.csv',index=False)
    dump(out/'completion.json',dict(baseline_files_unchanged=protected(),target_used_for_training=False,
        target_is_already_studied_dataset=True,interpretation='Exploratory domain-transfer assessment; no physical size mapping or independent target-acquisition claim.'))
    print(pd.DataFrame(rows).to_string(index=False),flush=True)

def tfm_reconstruct(data,xs,zs,fs=20000000,c=5820,pitch=.0007,time_zero=0):
    """Analytic full-aperture homogeneous TFM, trace order [time,tx,rx]."""
    n=data.shape[1];element=np.arange(n)*pitch
    xx,zz=np.meshgrid(xs,zs);points=np.stack([xx.ravel(),zz.ravel()],axis=1)
    analytic=hilbert(data,axis=0).astype(np.complex64);result=np.zeros(len(points),np.complex128)
    for i in range(n):
        dt=np.sqrt((points[:,0]-element[i])**2+points[:,1]**2)/c
        for j in range(n):
            t=(dt+np.sqrt((points[:,0]-element[j])**2+points[:,1]**2)/c-time_zero)*fs
            k=np.floor(t).astype(int);fraction=t-k;valid=(k>=0)&(k<data.shape[0]-1)
            trace=analytic[:,i,j]
            result[valid]+=(1-fraction[valid])*trace[k[valid]]+fraction[valid]*trace[k[valid]+1]
        if i%16==0:print('TFM transmitters:',i,'/',n,flush=True)
    return np.abs(result.reshape(xx.shape))/(n*n)

def measured_tfm(path):
    out=ROOT/'results/fmc_tfm';out.mkdir(exist_ok=True)
    if (out/'localisation.json').exists():raise RuntimeError('Completed TFM exists')
    raw=loadmat(path)['FMC_new'];assert raw.shape==(10000,128,128)
    # Metadata's stated axis order is corrected by inspecting MATLAB dimensions.
    sos=butter(4,[2e6,8e6],btype='bandpass',fs=100e6,output='sos')
    data=np.empty((2000,128,128),np.float32)
    for i in range(128):
        block=raw[:,i,:].astype(float);block-=block.mean(0,keepdims=True)
        data[:,i,:]=resample_poly(sosfiltfilt(sos,block,axis=0),1,5,axis=0)
    del raw
    xs=np.arange(0,90.001,.25)/1000;zs=np.arange(1,26.001,.25)/1000
    amplitude=tfm_reconstruct(data,xs,zs)
    np.savez_compressed(out/'tfm_image.npz',amplitude=amplitude,x_m=xs,z_m=zs)
    xx,zz=np.meshgrid(xs*1000,zs*1000);mask=(xx>=29)&(xx<=41)&(zz>=17)&(zz<=21.5)
    peak=np.unravel_index(np.where(mask,amplitude,-np.inf).argmax(),amplitude.shape)
    estimate=[float(xx[peak]),float(zz[peak])];truth=[35.,20.]
    result=dict(config=PLAN['tfm'],raw_sha256=sha(path),mat_variable='FMC_new',actual_mat_shape=[10000,128,128],
        estimate_x_z_mm=estimate,nominal_x_z_mm=truth,euclidean_error_mm=float(np.linalg.norm(np.array(estimate)-truth)),
        absolute_x_error_mm=abs(estimate[0]-35),absolute_depth_error_mm=abs(estimate[1]-20),
        caveat='ROI-assisted brightest indication, approximate metadata point, homogeneous weld approximation; not blind localisation or independent sizing.',
        dataset_doi='10.15129/a7da5071-0436-4913-8f0e-de78f5ebccd6',licence='CC BY 4.0',baseline_files_unchanged=protected())
    dump(out/'localisation.json',result)
    db=20*np.log10(np.maximum(amplitude/amplitude.max(),1e-8))
    fig,ax=plt.subplots(figsize=(8,4),layout='constrained');im=ax.pcolormesh(xs*1000,zs*1000,db,vmin=-50,vmax=0,cmap='inferno',shading='auto')
    ax.scatter([35],[20],marker='+',s=110,color='cyan',label='Approx. metadata lower tip')
    ax.scatter([estimate[0]],[estimate[1]],marker='x',s=65,color='lime',label='ROI peak')
    from matplotlib.patches import Rectangle
    ax.add_patch(Rectangle((29,17),12,4.5,fill=False,color='cyan',ls='--',lw=.8))
    ax.set(xlabel='Array-axis x (mm)',ylabel='Depth z (mm)',title='Measured full-matrix TFM | homogeneous longitudinal model');ax.invert_yaxis();ax.legend(fontsize=8)
    fig.colorbar(im,ax=ax,label='Relative amplitude (dB)');fig.savefig(out/'tfm_reconstruction.png',dpi=300);fig.savefig(out/'tfm_reconstruction.svg');plt.close(fig)
    print(json.dumps(result,indent=2),flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['transfer','tfm']);p.add_argument('--dataset-dir',type=Path);p.add_argument('--fmc',type=Path);p.add_argument('--resume-preprocessing',action='store_true');a=p.parse_args();setup();protected()
    if a.stage=='transfer':simulated_transfer(a.dataset_dir,a.resume_preprocessing)
    else:measured_tfm(a.fmc)

if __name__=='__main__':main()
