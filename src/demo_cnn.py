"""Portable inference demonstration on seven fixed, previously evaluated examples.

No raw data, training, GPU, or local preprocessing cache is required.
Run: python src/demo_cnn.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from dsp_cnn import CompactCNN

ROOT=Path(__file__).resolve().parents[1]

def main():
    torch.set_num_threads(2)
    data=np.load(ROOT/'data/cnn_demo_samples.npz')
    print('ELEC5305 | DSP + CNN saved-model demonstration')
    print('Previously evaluated test examples, selected by first sample ID per source; not a new test set.')
    print('Input: ROI -> DC removal -> Hilbert envelope -> block max -> log1p (already cached).')
    print('Loading fold-specific training normalization and seed-42 weights; threshold = 0.5.')
    rows=[]
    for i,sample_id in enumerate(data['sample_ids']):
        fold=int(data['folds'][i]);path=ROOT/f'results/dsp_cnn/source_holdout_{fold}_seed_42.pt'
        checkpoint=torch.load(path,map_location='cpu',weights_only=False)
        model=CompactCNN();model.load_state_dict(checkpoint['state_dict']);model.eval()
        x=torch.from_numpy((data['maps'][i:i+1,None]-checkpoint['mean'])/checkpoint['std'])
        with torch.no_grad():prob=float(torch.sigmoid(model(x))[0])
        rows.append(dict(sample_id=str(sample_id),fold=fold,true_label=int(data['labels'][i]),
            probability=round(prob,6),prediction=int(prob>=0.5)))
    print(pd.DataFrame(rows).to_string(index=False))
    print('For full six-fold / three-seed metrics, read results/dsp_cnn/cnn_summary.csv.')
    print('Limitation: all experimental negative rows repeat one background.')
    return rows

if __name__=='__main__':main()
