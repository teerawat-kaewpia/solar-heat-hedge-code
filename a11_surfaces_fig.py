"""Step 11 (v8): Fig. B1 - warming x PV-scale surfaces for all six load-model families."""
import pandas as pd, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
A = pd.read_csv('tab_surfaces_6families.csv')
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8.5})
names = list(dict.fromkeys(A.model)); short = ['GBM, monotone (reference)', 'GBM, short lags', 'GBM, unconstrained',
                                             'Spline additive', 'Linear hourly CDH', 'Hourly change-point']
f, ax = plt.subplots(2, 3, figsize=(10.5, 6.4), sharex=True, sharey=True, gridspec_kw={'hspace': 0.32, 'wspace': 0.08})
ax = ax.ravel(); lv = np.arange(-600, 601, 100)
rows = []
for i, (n, s) in enumerate(zip(names, short)):
    D = A[A.model == n]; base = D[(D.dT == 0) & (D.k == 1)].iloc[0]
    P = D.pivot(index='k', columns='dT', values='peak') - base.peak
    Sh = D.pivot(index='k', columns='dT', values='SHHR_hp') - base.SHHR_hp
    cf = ax[i].contourf(P.columns, P.index, P.values, levels=lv, cmap='RdBu_r', extend='both')
    ax[i].contour(P.columns, P.index, P.values, levels=[0], colors='k', linewidths=1.6)
    ax[i].contour(Sh.columns, Sh.index, Sh.values, levels=[0], colors='#1baf7a', linewidths=1.6, linestyles='--')
    ax[i].text(0.04, 5.75, s, fontsize=7.6, va='top', bbox=dict(fc='white', ec='none', alpha=0.8, pad=1.5))
    ax[i].set_title(f'({chr(97+i)})', loc='center', fontsize=10, fontweight='bold')
    ax[i].set_xticks([0, 1, 2, 3])
    if i >= 3: ax[i].set_xlabel('Warming ΔT (°C)')
    if i % 3 == 0: ax[i].set_ylabel('PV scale factor k (–)')
    for dT in [1.0, 2.0, 3.0]:
        col = P[dT]; kp = col.index[(col <= 0).values]; cs = Sh[dT]; ks = cs.index[(cs >= 0).values]
        rows.append(dict(model=s, dT=dT, kS_hp=float(ks.min()) if len(ks) else np.nan, kP=float(kp.min()) if len(kp) else np.nan))
cb = f.colorbar(cf, ax=ax, shrink=0.85, pad=0.015); cb.set_label('Change in net-peak proxy vs present (kW)')
ax[0].plot([], [], 'k-', lw=1.6, label='present net-peak proxy kept'); ax[0].plot([], [], '--', color='#1baf7a', lw=1.6, label='present SHHR$_{hp}$ kept')
ax[0].legend(fontsize=7, loc='center right', framealpha=0.85)
f.savefig('figB1.png', bbox_inches='tight', dpi=900)
print(pd.DataFrame(rows).pivot(index='model', columns='dT').round(2))
