"""Step 16 (revision): Fig. 5. (a) PV capacity multiplier needed under warming to hold annual grid imports (k*imp) and the
net-peak proxy (k*peak) at their present values, with the relative energy and peak shares as secondary checks. (b) The
net-peak proxy against the PV scale factor, with the present net-peak proxy, the point that defines
k*peak at +2 degC and the PV-free on-peak floor."""
import pandas as pd, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.grid': True, 'grid.color': '#e6e5e1', 'grid.linewidth': 0.6, 'axes.edgecolor': '#8a8984',
                     'legend.frameon': False, 'savefig.dpi': 900})
BLUE, ORANGE, INK, MUTED, GREY = '#2a78d6', '#eb6834', '#2b2a27', '#6b6a65', '#a8a7a2'
RAMP = ['#f2a07a', '#eb6834', '#b84a1f', '#7a2e12']          # ordinal ramp, validated (light -> dark = 0 -> +3 degC)
T = pd.read_csv('tab_like_for_like.csv'); Cv = pd.read_csv('tab_like_for_like_curves.csv')
r2 = T.set_index('dT').loc[2.0]
f, ax = plt.subplots(1, 2, figsize=(7.4, 3.3), gridspec_kw={'width_ratios': [1, 1.15]})
# (a) absolute targets first and emphasised, relative targets as thin grey secondary checks
series = [('k_import', BLUE, '-', 2.6, 'Annual grid imports, $k^*_{imp}$'), ('k_peak', ORANGE, '-', 2.6, 'Net-peak proxy, $k^*_{peak}$'),
          ('k_SHHR_hp', GREY, '--', 1.1, 'Energy share (SHHR$_{hp}$)'), ('k_peak_ratio', GREY, ':', 1.3, 'Peak share (net/gross)')]
for col, c, ls, lw, lab in series:
    main = lw > 2
    ax[0].plot(T.dT, T[col], color=c, ls=ls, lw=lw, marker='o' if main else None, ms=4.5, mec='white', mew=1, zorder=3 if main else 2)
ax[0].axvline(2.0, color=MUTED, lw=0.8, ls='--', zorder=1); ax[0].axhline(1, color=MUTED, lw=0.8, ls=':')
ax[0].text(2.08, r2.k_peak, f'{r2.k_peak:.2f}', va='center', fontsize=8, color=INK, fontweight='bold')
ax[0].text(1.92, r2.k_import + 0.15, f'{r2.k_import:.2f}', va='bottom', ha='right', fontsize=8, color=INK, fontweight='bold')
ax[0].set_xlabel('Warming ΔT (°C)'); ax[0].set_ylabel('PV capacity multiplier required\nto hold the present value, $k^*$')
ax[0].set_xticks([0, 1, 2, 3]); ax[0].set_ylim(0.8, 6)
ax[0].legend([Line2D([], [], color=c, ls=ls, lw=lw) for _, c, ls, lw, _ in series], [l for *_, l in series],
             loc='upper left', fontsize=7.2, handlelength=3.0)
# (b) net-peak proxy against k, with the point that defines k*peak at +2 degC
present = T.present_peak.iloc[0]
for i, (dT, g) in enumerate(Cv.groupby('dT')):
    ax[1].plot(g.k, g.peak, color=RAMP[i], lw=2, label=f'ΔT = +{dT:.0f} °C' if dT else 'ΔT = 0 °C')
    fl = T.loc[T.dT == dT, 'floor'].iloc[0]
    ax[1].plot([5.55, 6.0], [fl, fl], color=RAMP[i], lw=2, ls=(0, (1, 1)))
ax[1].axhline(present, color=INK, lw=1, ls='--', label=f'Present net-peak proxy ({present:,.0f} kW)')
kp = r2.k_peak
ax[1].plot([kp, kp], [1860, present], color=INK, lw=0.8, ls=':')
ax[1].scatter([kp], [present], s=36, color=RAMP[2], edgecolor=INK, lw=0.8, zorder=5)
ax[1].annotate(f'$k^*_{{peak}}$ = {kp:.2f}\nat +2 °C', (kp, present), (1.25, 1990), fontsize=7.2, color=INK,
               arrowprops=dict(arrowstyle='-', color=INK, lw=0.6))
ax[1].text(5.45, T.floor.min(), 'PV-free on-peak floor', fontsize=7.0, color=MUTED, ha='right', va='center')
ax[1].set_xlim(1, 6); ax[1].set_ylim(1860, 2880)
ax[1].set_xlabel('PV scale factor $k$\n(PV capacity = $k$ × 2.51 MWp)'); ax[1].set_ylabel('Net-peak proxy (kW)')
ax[1].legend(loc='upper right', fontsize=7.2, ncol=2, columnspacing=1.0)
for i, a in enumerate(ax): a.set_title(f'({chr(97+i)})', fontsize=10, fontweight='bold')
f.tight_layout(); f.savefig('fig5_rev.png', bbox_inches='tight', dpi=900); f.savefig('fig5_rev.pdf', bbox_inches='tight'); plt.close(f)
print('saved fig5_rev')
