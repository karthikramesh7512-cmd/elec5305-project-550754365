"""Submission 2 figures from saved CSV outputs only; no model computation."""
from pathlib import Path
import csv
import hashlib

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter, StrMethodFormatter
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
ORDER = ['P41_01', 'P41_02', 'P41_04', 'P41_06_notch', 'P41_05', 'P41_03']
LABELS = ['P41_01\n(2 mm)', 'P41_02\n(3 mm)', 'P41_04\n(6 mm)',
          'P41_06_notch\n(6 mm notch)', 'P41_05\n(17 mm)', 'P41_03\n(26 mm)']
COLS = ['held_out_source', 'flaw_size', 'DSP_recall', 'SVM_recall',
        'DSP_balanced_accuracy', 'SVM_balanced_accuracy', 'DSP_ROC_AUC', 'SVM_ROC_AUC']
INPUTS = [ROOT/'results/svm/dsp_vs_svm.csv',
          ROOT/'results/dsp_baseline/score_distribution_by_source.csv',
          ROOT/'results/dsp_baseline/selected_parameters.csv']
digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
before = {p: digest(p) for p in (ROOT/'results').rglob('*') if p.is_file()}

# Copy literal CSV field values to retain the saved numeric precision.
with INPUTS[0].open(newline='', encoding='utf-8') as f:
    rows = {row['held_out_source']: row for row in csv.DictReader(f)
            if row['evaluation'].startswith('Source holdout ')}
assert set(rows) == set(ORDER)
with (OUT/'results_table_submission2.csv').open('w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=COLS, extrasaction='ignore')
    writer.writeheader()
    writer.writerows(rows[source] for source in ORDER)
table = pd.DataFrame([rows[s] for s in ORDER])

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9,
                     'axes.labelsize': 10, 'axes.titlesize': 11,
                     'svg.fonttype': 'none', 'axes.spines.top': False,
                     'axes.spines.right': False, 'axes.linewidth': .7,
                     'savefig.facecolor': 'white'})
DSP, SVM = '#526778', '#B6C4CF'

def save(fig, name):
    for ext in ['png', 'svg']:
        fig.savefig(OUT/f'{name}.{ext}', dpi=400)
    plt.close(fig)

def bars(metric, title, ylabel, name):
    fig, ax = plt.subplots(figsize=(7.2, 4.65))
    fig.subplots_adjust(left=.105, right=.985, bottom=.25, top=.83)
    x = np.arange(6)
    for offset, method, color in [(-.19, 'DSP', DSP), (.19, 'SVM', SVM)]:
        values = table[f'{method}_{metric}'].astype(float).to_numpy()
        rectangles = ax.bar(x+offset, values, .36, color=color, edgecolor='#293640',
                            linewidth=.65, label=method, zorder=3)
        for rect, value in zip(rectangles, values):
            text = f'{value*100:.2f}%' if value not in [0, .5, 1] else f'{value*100:.0f}%'
            ax.text(rect.get_x()+rect.get_width()/2, value+.017, text,
                    ha='center', va='bottom', fontsize=8)
    ax.set_xticks(x, LABELS, fontsize=8.4)
    ax.set(xlabel='Held-out source', ylabel=ylabel, ylim=(0, 1.14))
    ax.set_yticks(np.arange(0, 1.01, .2))
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.grid(axis='y', color='#DCE1E5', linewidth=.55, zorder=0)
    ax.tick_params(axis='x', length=0, pad=8)
    fig.suptitle(title, y=.965, fontweight='bold', fontsize=11)
    fig.legend(*ax.get_legend_handles_labels(), loc='upper center',
               bbox_to_anchor=(.55,.92), ncol=2, frameon=False)
    fig.text(.105, .045, 'Six-fold source-flaw holdout only. No-flaw rows repeat one raw image;\n'
             'specificity does not establish generalisation to novel no-flaw examples.',
             fontsize=8.2, linespacing=1.4)
    save(fig, name)

bars('recall', 'Recall on previously unseen flaw sources', 'Recall', 'figure_recall_by_source')
bars('balanced_accuracy', 'Balanced accuracy by held-out flaw source',
     'Balanced accuracy', 'figure_balanced_accuracy_by_source')

dist = pd.read_csv(INPUTS[1]).set_index('source')
selected = pd.read_csv(INPUTS[2])
selected = selected[selected.scheme.eq('source_holdout')].sort_values('fold')
assert len(selected) == 6 and selected.window.eq(1).all()
assert table.DSP_ROC_AUC.astype(float).eq(1).all()
sources = ['no_flaw'] + ORDER
stats = [dict(label=s, med=dist.loc[s,'median'], q1=dist.loc[s,'p25'],
              q3=dist.loc[s,'p75'], whislo=dist.loc[s,'minimum'],
              whishi=dist.loc[s,'maximum'], fliers=[]) for s in sources]
fig, ax = plt.subplots(figsize=(7.2, 6.2))
fig.subplots_adjust(left=.12, right=.76, bottom=.32, top=.87)
ax.bxp(stats, positions=np.arange(7), widths=.5, showfliers=False, patch_artist=True,
       boxprops=dict(facecolor=SVM, edgecolor=DSP, linewidth=.9),
       medianprops=dict(color='#172A39', linewidth=1.2),
       whiskerprops=dict(color=DSP, linewidth=.8), capprops=dict(color=DSP, linewidth=.8), zorder=3)
fold_sources = ['P41_01','P41_02','P41_03','P41_04','P41_05','P41_06_notch']
for row in selected.itertuples():
    failed = row.fold in [1,2]
    color = '#9A432A' if failed else '#66727D'
    ax.axhline(row.threshold, color=color, linewidth=1 if failed else .8,
               linestyle=(0,(5,3)) if failed else (0,(2,3)), zorder=2)
    ax.text(1.02, row.threshold, f'Fold {row.fold}: {fold_sources[row.fold-1]}\n'
            f'τ = {row.threshold:,.2f}', transform=ax.get_yaxis_transform(),
            ha='left', va='center', fontsize=7.8, color=color)
ax.set_xticks(np.arange(7), ['No flaw'] + LABELS,
              rotation=35, ha='right', fontsize=8.2)
ax.set(ylabel='DSP score (stored amplitude units)', ylim=(2300,16300), xlim=(-.6,6.6))
ax.set_yticks(np.arange(4000,16001,2000))
ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
ax.grid(axis='y', linewidth=.5, color='#E4E7EA', zorder=0)
ax.tick_params(axis='x', length=0)
fig.suptitle('DSP score distributions and validation-selected thresholds',
             fontsize=11, fontweight='bold', y=.975)
fig.text(.12,.917,'Saved W = 1 scores · boxes: IQR and median · whiskers: full range',fontsize=8.5)
fig.text(.12,.045,'DSP scores remain separable (ROC-AUC = 1.0 in every current evaluation).\n'
         'Threshold transfer fails for unseen 2 mm and 3 mm flaws (folds 1 and 2).\n'
         'Each line is a separate fold’s validation-selected threshold; labels identify its test source.\n'
         'All 3,558 no-flaw rows contain the same raw image.',fontsize=8.2,linespacing=1.45)
save(fig,'figure_dsp_score_thresholds')
assert all(digest(p) == h for p,h in before.items()), 'Existing results changed'
print('Saved six figure files and one summary CSV. Existing result hashes unchanged.')
