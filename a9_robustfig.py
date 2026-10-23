"""Step 9: robustness figure (Fig. 9): cross-model multipliers, baseline sensitivity, high-temperature response."""
import json, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
A7 = json.load(open('results_a7.json')); KF = pd.read_csv('tab_kstar_fine.csv')
C = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#4a3aa7', '#e87ba4']
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.grid': True, 'grid.color': '#e6e5e1', 'grid.linewidth': 0.6, 'legend.frameon': False})
f, ax = plt.subplots(2, 2, figsize=(9.0, 6.6), gridspec_kw={'hspace': 0.42, 'wspace': 0.28}); ax = ax.ravel()
cm = A7['cross_model_stress']; dts = [0, 1, 2, 3]
for i, (name, r) in enumerate(cm.items()):
    ks = [1] + [r[f'kS_hp_{d}.0'] for d in dts[1:]]; kp = [1] + [r[f'kP_{d}.0'] for d in dts[1:]]
    lw = 2.4 if 'reference' in name else 1.3
    ax[0].plot(dts, ks, color=C[i], lw=lw, marker='o', ms=3.5, label=name)
    ax[1].plot(dts, [np.nan if (v is None or (isinstance(v, float) and np.isnan(v))) else v for v in kp], color=C[i], lw=lw, marker='o', ms=3.5)
for a in ax[:2]:
    a.set_xlabel('Warming ΔT (°C)'); a.set_xticks(dts); a.axhline(1, color='#52514e', lw=0.8, ls=':')
ax[0].set_ylabel('k* to keep present SHHR$_{hp}$'); ax[1].set_ylabel('k* to keep present net-peak proxy')
ax[0].set_ylim(0.9, 6.2); ax[1].set_ylim(0.9, 6.2); ax[0].legend(fontsize=6.6, loc='upper left')
TN = A7['tn_sensitivity']; tns = sorted(float(t) for t in TN)
for key, lab, c in [('heat_share', 'Temperature-driven share (%)', C[1]), ('SHHR_pr', 'SHHR$_{pr}$ (%)', C[4]), ('SHHR_hp', 'SHHR$_{hp}$ (%)', C[2])]:
    ax[2].plot(tns, [TN[str(t)][key] for t in tns], color=c, lw=2, marker='o', ms=3.5, label=lab)
ax[2].axvline(float(json.load(open('results.json'))['TN']), color='#52514e', ls='--', lw=0.9); ax[2].set_xlabel('Neutral temperature T$_n$ (°C)'); ax[2].set_ylabel('Indicator (%)')
ax[2].set_ylim(0, 92); ax[2].legend(fontsize=6.8, loc='lower left', bbox_to_anchor=(0.0, 0.66), ncol=1)
ax2b = ax[2].inset_axes([0.66, 0.70, 0.31, 0.25])
ax2b.plot(tns, [TN[str(t)]['kS_hp_2'] for t in tns], color=C[2], marker='o', ms=2.5, lw=1.2)
ax2b.plot(tns, [TN[str(t)]['kP_2'] for t in tns], color=C[1], marker='o', ms=2.5, lw=1.2)
ax2b.set_ylim(0.8, 4.0); ax2b.set_title('k* at +2 °C', fontsize=7); ax2b.tick_params(labelsize=6); ax2b.grid(False)
cv = A7['curves']; T = cv['T']; tmax = A7['extrapolation']['train_max']['Tamb']
sty = {'GBM, monotone (reference)': ('-', 2.2), 'Hourly change-point': ('--', 1.4), 'Linear hourly cooling-degree': (':', 1.6)}
for j, grp in enumerate(['afternoon 12-18', 'evening 18-24']):
    for name, (ls, lw) in sty.items():
        ax[3].plot(T, cv['data'][name][grp], color=C[1] if j == 0 else C[0], ls=ls, lw=lw,
                   label=f"{name.split(',')[0].replace('Linear hourly cooling-degree','Linear CDH')}, {grp.split()[0]} ({grp.split()[1].replace('-', ':00–')}:00)")
t72 = A7['extrapolation']['train_max']['T72']   # lagged features reach their observed maximum first [rev B4]
ax[3].axvline(t72, color='#52514e', lw=0.9, ls='--'); ax[3].text(t72+0.4, 1430, f'max observed\n72-h mean ({t72:.1f} °C)', ha='left', va='center', fontsize=6.5)
ax[3].axvline(tmax, color='#52514e', lw=0.9); ax[3].text(tmax-0.4, 860, f'max observed\nT ({tmax:.1f} °C)', ha='right', va='center', fontsize=6.5)
ax[3].set_xlabel('All temperature features set to T (°C)'); ax[3].set_ylabel('Mean predicted demand (kW)'); ax[3].legend(fontsize=6, loc='upper left')
for i, a in enumerate(ax): a.set_title(f'({chr(97+i)})', loc='center', fontsize=10, fontweight='bold')
f.savefig('fig_robust.png', bbox_inches='tight', dpi=900); print('ok')
