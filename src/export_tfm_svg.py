"""Export compact SVGs from saved TFM grids; no raw-data processing or fitting.

Rasterise only the dense amplitude mesh, keeping labels and markers as vectors.
The original PNGs and all numerical outputs are unchanged.
"""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def main():
    out=Path(__file__).resolve().parents[1]/'results/fmc_tfm'
    for calibrated in [False,True]:
        grid=np.load(out/('calibrated_tfm_image.npz' if calibrated else 'tfm_image.npz'))
        result=json.loads((out/('calibrated_localisation.json' if calibrated else 'localisation.json')).read_text())
        amplitude=grid['amplitude'];db=20*np.log10(np.maximum(amplitude/amplitude.max(),1e-8))
        fig,ax=plt.subplots(figsize=(8,4),layout='constrained')
        im=ax.pcolormesh(grid['x_m']*1000,grid['z_m']*1000,db,vmin=-50,vmax=0,cmap='inferno',shading='auto',rasterized=True)
        ax.scatter([35],[20],marker='+',s=110,color='cyan',label='Approx. metadata lower tip')
        estimate=result['estimate_x_z_mm'];ax.scatter([estimate[0]],[estimate[1]],marker='x',s=65,color='lime',label='ROI peak')
        ax.add_patch(Rectangle((29,17),12,4.5,fill=False,color='cyan',ls='--',lw=.8))
        if calibrated:ax.axhline(22,color='white',ls=':',lw=.8,label='Known backwall depth')
        ax.set(xlabel='Array-axis x (mm)',ylabel='Depth z (mm)',title='Backwall-calibrated TFM | time offset 1.210 us' if calibrated else 'Measured full-matrix TFM | homogeneous longitudinal model')
        ax.invert_yaxis();ax.legend(fontsize=8);fig.colorbar(im,ax=ax,label='Relative amplitude (dB)')
        path=out/('calibrated_tfm.svg' if calibrated else 'tfm_reconstruction.svg')
        fig.savefig(path,dpi=300);plt.close(fig);print(path.name,path.stat().st_size)

if __name__=='__main__':main()
