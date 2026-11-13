"""Regenerate final descriptive figures from saved predictions; never train or tune."""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
ROOT=Path(__file__).resolve().parents[1]
DOC=ROOT/'docs/methodology'
pred=pd.read_csv(ROOT/'results/dsp_cnn/per_sample_results.csv')
fig,axes=plt.subplots(1,3,figsize=(10.2,3.8),sharex=True,sharey=True,layout='constrained')
audits=[]
for ax,seed in zip(axes,[42,43,44]):
    frame=pred[pred.scheme.eq('source_holdout')&pred.fold.eq(1)&pred.seed.eq(seed)]
    positive=frame[frame.has_flaw.eq(1)];negative=frame[frame.has_flaw.eq(0)]
    rng=np.random.default_rng(seed)
    for group,y,color in [(negative,0,'#42667c'),(positive,1,'#c47b30')]:
        ax.scatter(group.logit,y+rng.uniform(-.12,.12,len(group)),s=6,color=color,alpha=.22,rasterized=True)
    ax.axvline(0,color='#a23224',ls='--',lw=1.5,label='Frozen decision boundary')
    tp=int((positive.logit>=0).sum());fn=len(positive)-tp
    ax.set(title=f'Seed {seed}: {tp}/547 detected\nRecall {100*tp/547:.2f}%',yticks=[0,1],yticklabels=['No flaw\n(repeated)','2 mm flaw'],xlabel='CNN logit',ylim=(-.4,1.4))
    ax.grid(axis='x',alpha=.2)
    audits.append(dict(seed=seed,n_positive=len(positive),n_negative=len(negative),
        maximum_negative_logit=negative.logit.max(),minimum_positive_logit=positive.logit.min(),
        separation_gap=positive.logit.min()-negative.logit.max(),threshold=0,TP=tp,FN=fn,
        false_positives=int((negative.logit>=0).sum()),recall=tp/len(positive)))
fig.savefig(DOC/'cnn_score_transfer.png',dpi=300);fig.savefig(DOC/'cnn_score_transfer.svg');plt.close(fig)
pd.DataFrame(audits).to_csv(DOC/'cnn_threshold_audit.csv',index=False)
(DOC/'cnn_threshold_provenance.json').write_text(json.dumps(dict(input='results/dsp_cnn/per_sample_results.csv',
    input_sha256=hashlib.sha256((ROOT/'results/dsp_cnn/per_sample_results.csv').read_bytes()).hexdigest(),
    operation='Descriptive plot and score extrema from frozen fold-1 predictions; no inference, tuning or fitting',
    jitter='Visualization only; seeded vertical jitter does not change x/logit values'),indent=2))

fig,ax=plt.subplots(figsize=(10,3.1));ax.set(xlim=(0,10),ylim=(0,3.1));ax.axis('off')
boxes=[(.2,1.65,2.1,'Shared DSP front end\nROI 1100:3100\nDC removal + Hilbert'),(2.85,1.65,2.1,'Representation\n8 x 16 block max\nlog1p -> 60 x 125'),(5.5,1.65,2.1,'Training-only scaling\nOne mean and SD\nfor each fold'),(8.05,1.65,1.75,'CNN decision\n5,953 weights\nlogit >= 0'),(2.6,.15,5.25,'Convolutions: 8 -> Pool -> 16 -> Pool -> 32\nGlobal mean + max -> 64 values -> Linear logit')]
for x,y,w,label in boxes:
    ax.add_patch(FancyBboxPatch((x,y),w,1.05,boxstyle='round,pad=0.05',fc='#edf1f3',ec='#42667c',lw=1))
    ax.text(x+w/2,y+.525,label,ha='center',va='center',fontsize=9)
for a,b in [(2.35,2.75),(5,5.4),(7.65,7.95)]:ax.annotate('',xy=(b,2.175),xytext=(a,2.175),arrowprops=dict(arrowstyle='->',color='#42667c'))
ax.annotate('',xy=(8.925,1.58),xytext=(7.85,.68),arrowprops=dict(arrowstyle='->',color='#42667c'))
fig.savefig(DOC/'cnn_pipeline_final.png',dpi=300,bbox_inches='tight');fig.savefig(DOC/'cnn_pipeline_final.svg',bbox_inches='tight');plt.close(fig)

