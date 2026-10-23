"""Step 24 (revision): graphical abstract, one landscape panel sized for a 13 cm print width.
Annual PV energy (HEOR) -> real-time coverage (SHHR_pr) -> +2 °C multipliers k*imp versus k*peak across six model families,
with the mean diurnal profiles of PV output and temperature-driven load. Every number is read from this run's outputs."""
import json, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 7, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.edgecolor': '#8a8984', 'axes.linewidth': 0.6, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6})
INK, MUTED, GRID = '#2b2a27', '#6b6a65', '#e6e5e1'
PV, LOAD, BLUE, ORANGE = '#e3a21a', '#4a3aa7', '#2a78d6', '#eb6834'

# ---- numbers (same definitions as the manuscript values file)
RV = json.load(open('results_rev.json')); ip = RV['ind_point']
HEOR, SHHR = ip['HEOR'], ip['SHHR_pr']
A22 = json.load(open('results_a22.json'))
ki = [A22[n]['2.0']['k_import'] for n in A22]; kp = [A22[n]['2.0']['k_peak'] for n in A22]
H = pd.read_pickle('hosp2.pkl').dropna(subset=['E', 'dLh']); hh = H.index.hour + H.index.minute/60
prof = H.groupby(hh)[['E', 'dLh']].mean()
pkE = float(prof.E.idxmax()); top = prof.dLh[prof.dLh >= 0.95*prof.dLh.max()].index   # broad maximum: within 5% of the peak
lag_lo, lag_hi = float(top.min()) - pkE, float(top.max()) - pkE
night = RV['night_share']
r1 = lambda x: f'{x:.1f}'

f = plt.figure(figsize=(13/2.54, 5.6/2.54))
f.text(0.012, 0.965, 'Annual energy balance overstates rooftop PV’s ability to limit hospital peak demand', fontsize=7.6, fontweight='bold', color=INK, va='top')
HY = 0.835   # step headers
def head(x, s_, ha='center'): f.text(x, HY, s_, fontsize=6.8, fontweight='bold', color=INK, ha=ha, va='center')

# ---- (left) diurnal profiles
head(0.215, 'Timing on a mean day')
ax = f.add_axes([0.062, 0.19, 0.30, 0.56])
x = np.r_[prof.index.values, 24.0]; e = np.r_[prof.E.values, prof.E.values[0]]/1000; d = np.r_[prof.dLh.values, prof.dLh.values[0]]/1000
ax.axvspan(0, 7.0, color='#efeeea', zorder=0, lw=0); ax.axvspan(18.5, 24, color='#efeeea', zorder=0, lw=0)   # night as in a4: before 07:00 or from 18:30
ax.fill_between(x, 0, d, color=LOAD, alpha=0.16, lw=0); ax.plot(x, d, color=LOAD, lw=1.3)
ax.plot(x, e, color=PV, lw=1.5)
ym = max(e.max(), d.max())*1.45; ax.set_ylim(0, ym)
ax.set_xlim(0, 24); ax.set_xticks([0, 6, 12, 18, 24])
ax.set_xlabel('Hour of day', fontsize=6.2, labelpad=1); ax.set_ylabel('Mean power (MW)', fontsize=6.2, labelpad=1)
ax.tick_params(labelsize=5.8, length=2, pad=1)
ax.text(8.6, e.max()*0.98, 'PV', color='#9a6a00', fontsize=6.4, ha='right', va='center', fontweight='bold')
ax.text(21.25, ym*0.50, 'Temperature-\ndriven load', color=LOAD, fontsize=5.8, ha='center', va='center', fontweight='bold')
ax.text(3.5, ym*0.45, f'night:\n{night:.0f}% of\nthis load', color=MUTED, fontsize=5.8, ha='center', va='center')
yl = e.max()*1.13
ax.add_patch(FancyArrowPatch((pkE, yl), (float(top.max()), yl), arrowstyle='-|>', mutation_scale=6, lw=0.8, color=INK))
ax.text((pkE + float(top.max()))/2, yl + ym*0.035, f'load peaks {lag_lo:.0f}–{lag_hi:.0f} h after PV', fontsize=5.6, ha='center', va='bottom', color=INK)

# ---- (middle) energy versus real-time coverage
def tile(x0, big, small, col, hd):
    head(x0, hd)
    f.text(x0, 0.56, big, fontsize=15, fontweight='bold', color=col, ha='center', va='center')
    f.text(x0, 0.33, small, fontsize=5.8, color=INK, ha='center', va='center', linespacing=1.2)
tile(0.462, f'{HEOR:.0f}%', 'annual PV energy\nrelative to the\ntemperature-driven\nenergy', INK, 'Annual balance')
tile(0.622, f'{SHHR:.0f}%', 'of the temperature-\ndriven load met by\nPV in the same\n30 min', LOAD, 'Real time')
f.add_artist(FancyArrowPatch((0.515, 0.56), (0.565, 0.56), transform=f.transFigure, arrowstyle='-|>', mutation_scale=8, lw=0.9, color=MUTED))
f.add_artist(FancyArrowPatch((0.682, 0.56), (0.722, 0.56), transform=f.transFigure, arrowstyle='-|>', mutation_scale=8, lw=0.9, color=MUTED))

# ---- (right) +2 °C multipliers
head(0.862, 'Under +2 °C warming')
ax2 = f.add_axes([0.745, 0.19, 0.235, 0.56])
rows = [(1, (min(ki), max(ki)), BLUE, '$k^*_{imp}$: keep annual grid imports'), (0, (min(kp), max(kp)), ORANGE, '$k^*_{peak}$: keep the net-peak proxy')]
for y, (lo, hi), col, lab in rows:
    ax2.plot([lo, hi], [y, y], color=col, lw=5.5, solid_capstyle='round', zorder=3)
    ax2.text(hi + 0.14, y, f'{r1(lo)}–{r1(hi)}×', fontsize=6.6, color=INK, va='center', fontweight='bold')
    ax2.text(1.0, y + 0.33, lab, fontsize=5.8, color=INK, va='center', ha='left')
ax2.set_ylim(-0.55, 1.75); ax2.set_xlim(1, 4.6); ax2.set_xticks([1, 2, 3, 4]); ax2.set_yticks([])
ax2.tick_params(labelsize=5.8, length=2, pad=1); ax2.spines['left'].set_visible(False)
ax2.set_xlabel('PV capacity needed (× present)', fontsize=6.0, labelpad=1)
ax2.grid(axis='x', color=GRID, lw=0.5, zorder=0)
ax2.text(4.6, 1.62, 'range over six load models', fontsize=5.8, color=MUTED, ha='right', va='center')

f.text(0.012, 0.022, 'Public hospital, northern Thailand  |  2.51 MWp rooftop PV  |  3.5 years of 30-min data', fontsize=5.8, color=MUTED, va='bottom')
f.savefig('graphical_abstract.png', dpi=600); f.savefig('graphical_abstract.pdf'); plt.close(f)
from PIL import Image   # TIFF without alpha channel, as preferred for journal artwork
Image.open('graphical_abstract.png').convert('RGB').save('graphical_abstract.tif', compression='tiff_lzw', dpi=(600, 600))
print('graphical abstract saved', dict(HEOR=round(HEOR, 1), SHHR_pr=round(SHHR, 1), lag=(lag_lo, lag_hi), night=round(night, 1),
      k_imp=(round(min(ki), 3), round(max(ki), 3)), k_peak=(round(min(kp), 3), round(max(kp), 3))))
