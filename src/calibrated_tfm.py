"""Separate backwall-calibrated TFM variant; preserve initial uncalibrated result."""
from pathlib import Path
import json
import numpy as np
from scipy.io import loadmat
from scipy.signal import butter,sosfiltfilt,resample_poly,hilbert
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from complete_extensions import ROOT,tfm_reconstruct,PLAN,dump,sha,protected

def main(path):
    out=ROOT/'results/fmc_tfm'
    if (out/'calibrated_localisation.json').exists():raise RuntimeError('Calibration experiment already completed')
    rule=dict(method='Median diagonal-pulse backwall arrival on remote aperture elements',
        plate_thickness_mm=22,sound_speed_m_s=5820,remote_elements_x_mm='x < 20 or x > 70',
        observed_arrival_window_us=[4.5,10.5],nominal_tip_used_in_calibration=False,
        explanation='Geometry-based time-origin calibration using a planar backwall; fixed before calibrated tip evaluation.')
    dump(out/'calibration_protocol.json',rule)
    raw=loadmat(path)['FMC_new'];assert raw.shape==(10000,128,128)
    sos=butter(4,[2e6,8e6],btype='bandpass',fs=100e6,output='sos')
    data=np.empty((2000,128,128),np.float32)
    for i in range(128):
        b=raw[:,i,:].astype(float);b-=b.mean(0,keepdims=True)
        data[:,i,:]=resample_poly(sosfiltfilt(sos,b,axis=0),1,5,axis=0)
    del raw
    fs=20e6;c=5820.;nominal=2*.022/c
    records=[]
    for i in range(128):
        x=i*.7
        if not (x<20 or x>70):continue
        env=np.abs(hilbert(data[:,i,i]));lo=int(4.5e-6*fs);hi=int(10.5e-6*fs)
        k=lo+int(env[lo:hi+1].argmax());observed=k/fs
        records.append(dict(element=i,x_mm=x,observed_backwall_us=observed*1e6,offset_us=(nominal-observed)*1e6))
    import pandas as pd
    pd.DataFrame(records).to_csv(out/'backwall_calibration.csv',index=False)
    offset=float(np.median([r['offset_us'] for r in records]))*1e-6
    xs=np.arange(0,90.001,.25)/1000;zs=np.arange(1,26.001,.25)/1000
    amplitude=tfm_reconstruct(data,xs,zs,time_zero=offset)
    np.savez_compressed(out/'calibrated_tfm_image.npz',amplitude=amplitude,x_m=xs,z_m=zs)
    xx,zz=np.meshgrid(xs*1000,zs*1000);mask=(xx>=29)&(xx<=41)&(zz>=17)&(zz<=21.5)
    k=np.unravel_index(np.where(mask,amplitude,-np.inf).argmax(),amplitude.shape)
    estimate=[float(xx[k]),float(zz[k])]
    result=dict(calibration=rule,time_zero_seconds=offset,nominal_lower_tip_x_z_mm=[35,20],estimate_x_z_mm=estimate,
        euclidean_error_mm=float(np.linalg.norm(np.array(estimate)-[35,20])),
        absolute_x_error_mm=abs(estimate[0]-35),absolute_depth_error_mm=abs(estimate[1]-20),
        data_sha256=sha(path),code_sha256=sha(Path(__file__)),baseline_files_unchanged=protected(),
        caveat='Metadata-assisted brightest peak in predeclared lower-tip ROI; nominal position approximate. No calibration by crack-tip error and no blind localisation or crack-size claim.')
    dump(out/'calibrated_localisation.json',result)
    db=20*np.log10(np.maximum(amplitude/amplitude.max(),1e-8));fig,ax=plt.subplots(figsize=(8,4),layout='constrained')
    im=ax.pcolormesh(xs*1000,zs*1000,db,vmin=-50,vmax=0,cmap='inferno',shading='auto')
    ax.scatter([35],[20],marker='+',s=110,color='cyan',label='Approx. metadata tip')
    ax.scatter([estimate[0]],[estimate[1]],marker='x',s=70,color='lime',label='ROI peak')
    ax.axhline(22,color='white',ls=':',lw=.8,label='Known backwall depth')
    ax.set(xlabel='Array-axis x (mm)',ylabel='Depth z (mm)',title=f'Backwall-calibrated TFM | time offset {offset*1e6:.3f} us');ax.invert_yaxis();ax.legend(fontsize=8)
    fig.colorbar(im,ax=ax,label='Relative amplitude (dB)');fig.savefig(out/'calibrated_tfm.png',dpi=300);fig.savefig(out/'calibrated_tfm.svg');plt.close(fig)
    print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--fmc',type=Path,required=True);main(p.parse_args().fmc)
