"""Render the first-half methodology figure as editable SVG and 400 dpi PNG."""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch

OUT = Path(__file__).resolve().parent
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                     'svg.fonttype': 'none', 'savefig.facecolor': 'white'})
fig, ax = plt.subplots(figsize=(7.2, 10.6))
fig.subplots_adjust(left=.025, right=.975, top=.985, bottom=.015)
ax.set(xlim=(0, 10), ylim=(0, 25.3))
ax.axis('off')
ink = '#243444'
edge = '#526474'
fill = '#F2F5F7'

def box(x, y, text, width=6.1, height=.66, bold=False, tint=False, size=10):
    ax.add_patch(Rectangle((x-width/2, y-height/2), width, height,
                           linewidth=.85, edgecolor=edge,
                           facecolor=fill if tint else 'white', zorder=3))
    ax.text(x, y, text, ha='center', va='center', color=ink, fontsize=size,
            fontweight='bold' if bold else 'normal', linespacing=1.25, zorder=4)
    return (x, y, width, height)

def line(points):
    ax.plot(*zip(*points), color=edge, linewidth=.9, zorder=1,
            solid_capstyle='butt', solid_joinstyle='miter')

def arrow(start, end):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle='-|>', mutation_scale=9,
                                linewidth=.9, color=edge, shrinkA=0, shrinkB=0,
                                zorder=2))

def connect(a, b):
    arrow((a[0], a[1]-a[3]/2), (b[0], b[1]+b[3]/2))

def split(a, left, right, junction):
    line([(a[0], a[1]-a[3]/2), (a[0], junction)])
    line([(left[0], junction), (right[0], junction)])
    for b in (left, right):
        arrow((b[0], junction), (b[0], b[1]+b[3]/2))

def merge(left, right, b, junction):
    for a in (left, right):
        line([(a[0], a[1]-a[3]/2), (a[0], junction)])
    line([(left[0], junction), (right[0], junction)])
    arrow((b[0], junction), (b[0], b[1]+b[3]/2))

a = box(5, 24.7, 'NDT_ML_Flaw Dataset', bold=True, tint=True)
b = box(5, 23.52, 'Experimental batches 013–019\n7000 samples', height=.94)
c = box(5, 22.13, 'Dataset and Metadata Audit\n6 experimental flaw sources', height=.94)
d = box(5, 20.95, 'Evaluation Protocol', bold=True, tint=True)
for first, second in [(a,b),(b,c),(c,d)]: connect(first, second)

l = box(2.45, 19.22, 'Ordinary image-level split\n70/15/15\nReference evaluation', width=4.5, height=1.45)
r = box(7.55, 19.22, 'Six-fold source-flaw holdout\nTest flaw source absent from training\nMain generalisation evaluation', width=4.5, height=1.45, size=9.2)
split(d, l, r, 20.32)
roi = box(5, 17.52, 'Fixed ROI: columns 1100:3100')
merge(l, r, roi, 18.12)
dc = box(5, 16.43, 'Per-A-scan DC removal')
env = box(5, 15.34, 'Hilbert envelope')
connect(roi, dc)
connect(dc, env)

left_labels = ['Deterministic DSP', 'Maximum line response', 'Spatial smoothing',
               'Validation-selected threshold', 'DSP prediction']
right_labels = ['Engineered feature extraction', '12 interpretable DSP features',
                'StandardScaler', 'SVM', 'SVM prediction']
ys = [13.93, 12.78, 11.63, 10.48, 9.33]
left = [box(2.45, y, label, width=4.5, height=.7, bold=i==0, tint=i==0,
            size=9.7) for i, (y, label) in enumerate(zip(ys, left_labels))]
right = [box(7.55, y, label, width=4.5, height=.7, bold=i==0, tint=i==0,
             size=9.7) for i, (y, label) in enumerate(zip(ys, right_labels))]
split(env, left[0], right[0], 14.73)
for branch in [left, right]:
    for first, second in zip(branch, branch[1:]): connect(first, second)

test = box(5, 7.88, 'Common held-out test sets')
merge(left[-1], right[-1], test, 8.67)
comparison = box(5, 6.72, 'DSP vs SVM comparison', bold=True, tint=True)
connect(test, comparison)
analysis = box(5, 4.66, 'Generalisation analysis by:\n• source flaw\n• flaw size\n• recall\n• balanced accuracy', height=2.45)
connect(comparison, analysis)
ax.text(5, 2.1, 'Test data are not used for parameter or hyperparameter selection.',
        ha='center', va='center', fontsize=9, color=ink)

# Keep a compact report figure while preserving the physical text size.
ax.set_ylim(1.35, 25.3)
for ext in ['svg', 'png']:
    fig.savefig(OUT / f'methodology_pipeline.{ext}', dpi=400)
plt.close(fig)
