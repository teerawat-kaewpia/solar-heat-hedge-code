"""Step 23 (revision): Fig. 6, robustness of the headline energy versus peak comparison across the six model families,
and Fig. S6, neutral-temperature sensitivity and the high-temperature response (formerly panels c and d of Fig. 6)."""
import json, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.grid': True, 'grid.color': '#e6e5e1', 'grid.linewidth': 0.6, 'axes.edgecolor': '#8a8984', 'legend.frameon': False})
BLUE, ORANGE, INK, GREY = '#2a78d6', '#eb6834', '#2b2a27', '#a8a7a2'
A22 = json.load(open('results_a22.json'))
SHORT = {'GBM, monotone (reference)': 'GBM, monotone (reference)', 'GBM, short lags only': 'GBM, short lags',
         'GBM, unconstrained': 'GBM, unconstrained', 'Spline additive (GAM-like)': 'Additive spline',
         'Linear hourly cooling-degree': 'Linear cooling-degree', 'Hourly change-point': 'Hourly change-point'}
names = list(A22)
# ---------------- Fig. 6 ----------------
f, ax = plt.subplots(1, 2, figsize=(7.4, 3.3), gridspec_kw={'width_ratios': [1.15, 1], 'wspace': 0.42})
y = np.arange(len(names))[::-1]
for yi, n in zip(y, names):
    r = A22[n]['2.0']; ki, kp = r['k_import'], r['k_peak']
    ax[0].plot([ki, kp], [yi, yi], color=GREY, lw=1.2, zorder=1)
    ax[0].scatter([ki], [yi], s=34, color=BLUE, edgecolor='white', lw=0.8, zorder=3, label='$k^*_{imp}$ (annual grid imports)' if yi == y[0] else None)
    ax[0].scatter([kp], [yi], s=34, color=ORANGE, edgecolor='white', lw=0.8, zorder=3, label='$k^*_{peak}$ (net-peak proxy)' if yi == y[0] else None)
    ax[0].text(3.62, yi, f'×{kp/ki:.1f}', va='center', ha='left', fontsize=7.4, color=INK)
ax[0].set_yticks(y); ax[0].set_yticklabels([SHORT[n] for n in names], fontsize=7.4)
ax[0].set_xlim(1, 3.55); ax[0].set_xlabel('PV capacity multiplier at +2 °C, $k^*$'); ax[0].grid(axis='y', visible=False)
ax[0].legend(loc='upper center', bbox_to_anchor=(0.45, -0.20), ncol=2, fontsize=7.2, handletextpad=0.3, columnspacing=1.0)
ax[0].text(3.62, y[0] + 0.55, r'$k^*_{peak}/k^*_{imp}$', fontsize=6.6, color=INK, ha='left', fontweight='bold')
dts = [0.0] + [float(d) for d in A22[names[0]]]
for n in names:
    ratio = [1.0] + [A22[n][f'{d:.1f}']['ratio'] for d in dts[1:]]
    ref = 'reference' in n
    ax[1].plot(dts, ratio, color=INK if ref else GREY, lw=2.2 if ref else 1.1, marker='o' if ref else None, ms=3.5, zorder=3 if ref else 2)
    if ref or n.startswith('Spline'):
        ax[1].text(3.06, ratio[-1], SHORT[n].replace(' (reference)', '\n(reference)'), fontsize=6.6, va='center', color=INK)
ax[1].axhline(1, color=INK, lw=0.8, ls=':'); ax[1].text(2.95, 0.97, 'equal multipliers', fontsize=6.4, color=INK, va='top', ha='right')
ax[1].set_xlim(0, 3.0); ax[1].set_xticks([0, 1, 2, 3]); ax[1].set_ylim(0.9, 3.3)
ax[1].set_xlabel('Warming ΔT (°C)'); ax[1].set_ylabel('$k^*_{peak}$ / $k^*_{imp}$')
ax[1].text(0.08, 3.18, 'grey: five alternative families', fontsize=6.4, color='#6b6a65', va='top')
for i, a in enumerate(ax): a.set_title(f'({chr(97+i)})', fontsize=10, fontweight='bold')
f.savefig('fig6_rev.png', bbox_inches='tight', dpi=900); f.savefig('fig6_rev.pdf', bbox_inches='tight'); plt.close(f)
# ---------------- Fig. S6 ----------------
A7 = json.load(open('results_a7.json')); TN = A7['tn_sensitivity']; tns = sorted(float(t) for t in TN)
f, ax = plt.subplots(1, 2, figsize=(7.4, 3.2), gridspec_kw={'wspace': 0.32})
for key, lab, c in [('heat_share', 'Temperature-driven share', ORANGE), ('SHHR_pr', 'SHHR$_{pr}$', '#4a3aa7'), ('SHHR_hp', 'SHHR$_{hp}$', '#1baf7a')]:
    ax[0].plot(tns, [TN[str(t)][key] for t in tns], color=c, lw=2, marker='o', ms=3.5, label=lab)
ax[0].axvline(float(json.load(open('results.json'))['TN']), color='#52514e', ls='--', lw=0.9)
ax[0].set_xlabel('Neutral temperature $T_n$ (°C)'); ax[0].set_ylabel('Indicator (%)'); ax[0].set_ylim(0, 70); ax[0].legend(fontsize=7, loc='upper right')
cv = A7['curves']; T = cv['T']; tmax = A7['extrapolation']['train_max']['Tamb']; t72 = A7['extrapolation']['train_max']['T72']
sty = {'GBM, monotone (reference)': ('-', 2.2), 'Hourly change-point': ('--', 1.4), 'Linear hourly cooling-degree': (':', 1.6)}
for j, grp in enumerate(['afternoon 12-18', 'evening 18-24']):
    for name, (ls, lw) in sty.items():
        ax[1].plot(T, cv['data'][name][grp], color=ORANGE if j == 0 else BLUE, ls=ls, lw=lw,
                   label=f"{name.split(',')[0].replace('Linear hourly cooling-degree', 'Linear CDH')}, {grp.split()[0]} ({grp.split()[1].replace('-', ':00–')}:00)")
ax[1].axvspan(t72, max(T), color='#efeeea', zorder=0)
ax[1].axvline(t72, color='#52514e', lw=0.9, ls='--'); ax[1].text(t72 + 0.4, 1520, f'beyond the highest\n72-h mean\n({t72:.1f} °C)', fontsize=6.0, va='center')
ax[1].axvline(tmax, color='#52514e', lw=0.9); ax[1].text(tmax + 0.4, 1000, f'highest\nobserved T\n({tmax:.1f} °C)', ha='left', fontsize=6.0, va='center')
ax[1].set_xlabel('All temperature features set to T (°C)'); ax[1].set_ylabel('Mean predicted demand (kW)'); ax[1].legend(fontsize=5.8, loc='upper left')
for i, a in enumerate(ax): a.set_title(f'({chr(97+i)})', fontsize=10, fontweight='bold')
f.savefig('figS6.png', bbox_inches='tight', dpi=900); plt.close(f)
print('fig6_rev and figS6 saved')
