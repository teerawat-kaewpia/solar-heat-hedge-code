"""Step 3: derived metrics (phase lag, hedge-preserving multipliers) + publication figures."""
import pandas as pd, numpy as np, json, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
R = json.load(open('results.json'))
H = pd.read_pickle('hosp2.pkl'); Dd = pd.read_pickle('daily.pkl')
G = pd.read_csv('stress_grid.csv'); Bt = pd.read_csv('battery.csv'); pk = pd.read_csv('peaks_monthly.csv')
prof = pd.read_csv('prof_regime.csv'); st = pd.read_csv('tab_strat.csv')
C = dict(b='#2a78d6', o='#eb6834', a='#1baf7a', y='#eda100', v='#4a3aa7', g='#52514e', m='#e87ba4')
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.grid': True, 'grid.color': '#e6e5e1', 'grid.linewidth': 0.6, 'axes.edgecolor': '#8a8984',
                     'axes.titlesize': 9.5, 'axes.titleweight': 'bold', 'legend.frameon': False, 'savefig.dpi': 900})
X = {}
# ---------- phase lag between heat-driven load and PV (cross-correlation, clear+hot days) ----------
V = H.dropna(subset=['E','dLh']).copy()
hh = V.index.hour + V.index.minute/60
pE = V.groupby(hh).E.mean(); pD = V.groupby(hh).dLh.mean()
X['centroid_E'] = float((pE.index*pE).sum()/pE.sum()); X['centroid_dLh_day'] = float((pD[(pD.index>=6)&(pD.index<19)].index*pD[(pD.index>=6)&(pD.index<19)]).sum()/pD[(pD.index>=6)&(pD.index<19)].sum())
lags = range(-8, 9); cc = []
Vg = V[['E','dLh']].reindex(pd.date_range(H.index.min(), H.index.max(), freq='30min'))   # regular grid so shift() = 30 min [rev A4]
e = Vg.E - Vg.E.rolling('24h', center=True, min_periods=48).mean(); d = Vg.dLh - Vg.dLh.rolling('24h', center=True, min_periods=48).mean()
for l in lags: cc.append(e.corr(d.shift(-l)))
X['lag_best_h'] = float(list(lags)[int(np.nanargmax(cc))]*0.5); X['cc_max'] = float(np.nanmax(cc)); X['cc0'] = float(cc[8])
X['peak_hour_E'] = float(pE.idxmax()); X['peak_hour_dLh'] = float(pD.idxmax())   # interval-end time [rev A2]
X['night_heat_share'] = float(V.loc[(hh < 7) | (hh >= 18.5), 'dLh'].sum()/V.dLh.sum()*100)
# uncovered heat load by hour
V['unc'] = V.dLh - np.minimum(np.minimum(V.E, V.L), V.dLh); unc = V.groupby(hh)[['dLh','unc']].sum()/V.dLh.sum()*100
# ---------- hedge- and peak-preserving PV multipliers ----------
base = G[(G.dT==0)&(G.k==1)].iloc[0]
kstar = {}
for dT, g in G.groupby('dT'):
    g = g[g.k > 0].sort_values('k')
    ks = np.interp(base.SHHR, g.SHHR, g.k, right=np.nan)
    kp = np.interp(-base.peak_net, -g.peak_net.values, g.k.values, right=np.nan)
    kstar[float(dT)] = dict(k_SHHR=float(ks), k_peak=float(kp) if not np.isnan(kp) else None,
                            SHHR_k1=float(g[g.k==1].SHHR.iloc[0]), peak_k1=float(g[g.k==1].peak_net.iloc[0]),
                            L=float(g.L_GWh.iloc[0]), dLh=float(g.dLh_GWh.iloc[0]))
import os
if os.path.exists('tab_kstar_fine.csv'):          # fine-grid (dk = 0.05) multipliers from a8_finegrid.py
    KF = pd.read_csv('tab_kstar_fine.csv').set_index('dT')
    for dT in kstar:
        if dT in KF.index:
            kstar[dT]['k_SHHR'] = float(KF.loc[dT, 'kS_hp'])
            kstar[dT]['k_peak'] = None if np.isnan(KF.loc[dT, 'kP']) else float(KF.loc[dT, 'kP'])
X['kstar'] = kstar
g0 = G[G.k==0].set_index('dT'); X['load_per_degC_GWh'] = float(np.polyfit(g0.index, g0.L_GWh, 1)[0])
X['dLh_per_degC_pct'] = float(np.polyfit(g0.index, g0.dLh_GWh, 1)[0]/g0.dLh_GWh.iloc[0]*100)
g1 = G[G.k==1].set_index('dT'); X['SHHR_per_degC'] = float(np.polyfit(g1.index, g1.SHHR, 1)[0])
X['E_per_degC_pct'] = float(np.polyfit(g1.index, g1.E_GWh, 1)[0]/g1.E_GWh.iloc[0]*100)
json.dump(X, open('results_extra.json','w'), indent=1); print(json.dumps(X, indent=1))

# ======================= FIGURES =======================
def save(f, n):
    axs = [a for a in f.axes if a.get_label() != '<colorbar>']
    for i, a in enumerate(axs):
        a.set_title('', loc='left'); a.set_title('', loc='right')
        a.set_title(f'({chr(97+i)})' if len(axs) > 1 else '', loc='center', fontsize=10, fontweight='bold')
    f.savefig(f'fig{n}.png', bbox_inches='tight', dpi=900); plt.close(f)
# Fig 1 framework: one reading direction, the energy versus capacity comparison as the end point [rev]
f, ax = plt.subplots(figsize=(7.4, 3.1)); ax.axis('off'); ax.set_xlim(0, 10.6); ax.set_ylim(0, 4.3)
INK = '#2b2a27'; GREY = '#8a8984'
cards = [  # x, width, title, body, colour, emphasis
    (0.05, 1.78, '1  Data and\nquality control', '3.5 years,\n30-min intervals\n\nThree PV zones,\n2.51 MWp\n\nCounter\nreconstruction and\nclosure check\n\nInvoice-verified\nCT correction', C['b'], False),
    (2.05, 1.98, '2  Temperature-load\nattribution', 'Daily change-point\nmodel\n\nMonotone 30-min\nload model\n\nNeutral-temperature\ncounterfactual\n\n$\\Delta L_{heat}=f(T)-f(\\min[T,T_n])$', C['b'], False),
    (4.25, 2.12, '3  Observed\nenergy-peak mismatch', 'Annual energy\nbalance (HEOR)\n\nCoincident PV\ncoverage (SHHR)\n\nPeak reduction\nand timing\n\nSolar-window gap', C['v'], False),
    (6.59, 1.66, '4  Warming × PV\nstress test', 'ΔT = 0–3 °C\n\nPV scale\nk = 1–6\n\nZero export,\nno storage', C['o'], False),
    (8.47, 2.08, '5  Preservation\ntargets', None, C['v'], True)]
y0, h = 0.95, 3.25
for x, w, title, body, col, strong in cards:
    ax.add_patch(FancyBboxPatch((x, y0), w, h, boxstyle='round,pad=0.03,rounding_size=0.10',
                                fc=col + ('33' if strong else '14'), ec=col, lw=2.6 if strong else 1.2))
    ax.text(x + w/2, y0 + h - 0.12, title, ha='center', va='top', fontsize=6.0, fontweight='bold', color=INK, linespacing=1.05)
    if body: ax.text(x + w/2, y0 + h - 0.68, body, ha='center', va='top', fontsize=5.8, color=INK, linespacing=1.05)
x5, w5 = cards[-1][0], cards[-1][1]
for k, (head, what, sym) in enumerate([('Energy\ntarget', 'Annual grid\nimports', '$k^*_{imp}$'), ('Capacity\ntarget', 'Net-peak\nproxy', '$k^*_{peak}$')]):
    cx = x5 + w5*(0.26 + 0.48*k)
    ax.text(cx, y0 + 2.52, head, ha='center', va='top', fontsize=5.8, fontweight='bold', color=INK, linespacing=1.05)
    ax.text(cx, y0 + 1.95, what, ha='center', va='top', fontsize=5.8, color=INK, linespacing=1.05)
    ax.text(cx, y0 + 1.22, sym, ha='center', va='top', fontsize=10, color=INK)
ax.plot([x5 + w5/2, x5 + w5/2], [y0 + 0.62, y0 + 2.50], color=C['v'], lw=0.8)
ax.text(x5 + w5/2, y0 + 0.30, 'Relative targets:\nsecondary checks', ha='center', va='center', fontsize=5.5, color=INK, linespacing=1.05)
for (x, w, *_), (xn, *_) in zip(cards[:-1], cards[1:]):
    ax.annotate('', (xn - 0.02, y0 + h/2), (x + w + 0.02, y0 + h/2), arrowprops=dict(arrowstyle='-|>', color='#52514e', lw=1.1))
ax.add_patch(FancyBboxPatch((0.05, 0.05), 10.5, 0.62, boxstyle='round,pad=0.02,rounding_size=0.08', fc='#efeeea', ec='#c9c8c3', lw=0.8))
ax.text(5.3, 0.36, 'Validation and uncertainty:   invoice check   |   temporal holdout   |   six model families   |   '
        'season-stratified bootstrap   |   metering sensitivity', ha='center', va='center', fontsize=5.4, color=INK)
save(f, 1)
# Fig 2 data overview (two panels, one axis each)
Dm = Dd.copy(); roll = Dm[['L','E','Tmax']].rolling('14D', center=True, min_periods=7).mean()   # calendar window [rev A4]
f, ax = plt.subplots(2, 1, figsize=(7.2, 4.6), sharex=True, gridspec_kw={'height_ratios': [1.6, 1]})
ax[0].plot(Dm.index, Dm.L/1e3, color=C['b'], lw=0.5, alpha=0.35); ax[0].plot(roll.index, roll.L/1e3, color=C['b'], lw=1.8, label='Gross hospital load, L')
ax[0].plot(Dm.index, Dm.E/1e3, color=C['o'], lw=0.5, alpha=0.35); ax[0].plot(roll.index, roll.E/1e3, color=C['o'], lw=1.8, label='Rooftop PV generation, E')
ax[0].set_ylabel('Daily energy (MWh d$^{-1}$)'); ax[0].legend(loc='upper left', ncol=2); ax[0].set_title('(a) Daily energy, 14-day centred mean in bold', loc='left')
ax[1].plot(Dm.index, Dm.Tmax, color=C['g'], lw=0.5, alpha=0.4); ax[1].plot(roll.index, roll.Tmax, color='#0b0b0b', lw=1.6)
hw = Dm[Dm.HW]; ax[1].scatter(hw.index, hw.Tmax, s=9, color=C['o'], zorder=3, label='Heatwave day (≥3 d, T$_{max}$ ≥ P90)')
ax[1].set_ylabel('T$_{max}$ (°C)'); ax[1].legend(loc='lower right'); ax[1].set_title('(b) Daily maximum ambient temperature', loc='left')
save(f, 2)
# Fig 3 thermal response
cp = R['cp_hospital']; f, ax = plt.subplots(1, 2, figsize=(7.2, 3.1))
for hol, c, lab in [(False, C['b'], 'Working day'), (True, C['o'], 'Weekend / holiday')]:
    s = Dd[Dd.hol == hol]; ax[0].scatter(s['T'], s.L/1e3, s=6, color=c, alpha=0.45, lw=0, label=lab)
    t = np.linspace(14, 33, 100); y = cp['b0'] + cp['bhol']*hol + cp['slope']*np.clip(t-cp['Tb'], 0, None)
    ax[0].plot(t, y/1e3, color=c, lw=2)
ax[0].axvline(cp['Tb'], color=C['g'], ls='--', lw=1); ax[0].text(cp['Tb']-0.3, 33, f"T$_b$ = {cp['Tb']:.1f} °C", fontsize=8, ha='right')
ax[0].set_xlabel('Daily mean ambient temperature (°C)'); ax[0].set_ylabel('Daily gross load (MWh d$^{-1}$)')
ax[0].legend(loc='upper left', markerscale=2); ax[0].set_title('(a) 3P change-point model (daily)', loc='left')
pdT = np.array(R['pdT']); ax[1].plot(pdT[:,0], pdT[:,1], color=C['v'], lw=2)
ax[1].axvline(R['TN'], color=C['g'], ls='--', lw=1); ax[1].set_xlabel('Temperature features set to T (°C)'); ax[1].set_ylabel('Mean predicted demand (kW)')
t72max = float(H.T72.max())   # highest observed 72-h mean: the curve beyond it has limited support [rev]
ax[1].axvspan(t72max, pdT[:, 0].max(), color='#efeeea', zorder=0, lw=0); ax[1].axvline(t72max, color=C['g'], lw=0.8)
ax[1].text(t72max + 0.4, pdT[:, 1].min() + 0.05*(pdT[:, 1].max() - pdT[:, 1].min()), 'limited support\nbeyond the highest\nobserved 72-h mean', fontsize=6.4, va='bottom')
ax[1].set_title('(b) Monotone GBM, partial dependence', loc='left')
save(f, 3)
# Fig 4 diurnal profiles by regime
f, ax = plt.subplots(1, 3, figsize=(7.2, 2.9), sharey=True)
REG = {'cool (Tmax<30)': 'Cool: T$_{max}$ < 30 °C', 'warm (30-34)': 'Warm: 30–34 °C', 'hot (>=34)': 'Hot: T$_{max}$ ≥ 34 °C'}
for i, r in enumerate(['cool (Tmax<30)', 'warm (30-34)', 'hot (>=34)']):
    p = prof[prof.reg == r]
    ax[i].fill_between(p.hhf, p.L-p.dLh, p.L, color=C['o'], alpha=0.25, lw=0, label='Temperature-driven component')
    ax[i].plot(p.hhf, p.L, color=C['b'], lw=2, label='Gross load L'); ax[i].plot(p.hhf, p.B, color='#0b0b0b', lw=1.4, ls='--', label='Net import B')
    ax[i].plot(p.hhf, p.E, color=C['y'], lw=2, label='PV E'); ax[i].set_xticks([0, 6, 12, 18, 24]); ax[i].set_xlabel('Hour of day')
    ax[i].text(0.03, 0.98, REG[r], transform=ax[i].transAxes, ha='left', va='top', fontsize=7.4, fontweight='bold')
ax[0].set_ylabel('Mean working-day power (kW)'); ax[0].legend(loc='upper left', fontsize=6.2, bbox_to_anchor=(0.0, 0.90))
# solar window and lag, read from the warm-day profile itself (it matches the record-wide estimate in the text) [rev]
pw = prof[prof.reg == 'warm (30-34)'].set_index('hhf'); pv_pk = pw.E.idxmax(); top = pw.dLh[pw.dLh >= 0.95*pw.dLh.max()].index
ax[1].axvline(pv_pk, color=C['y'], lw=1.0, ls='--'); ax[1].axvspan(top.min(), top.max(), color=C['o'], alpha=0.12, lw=0)
ya = 300; ax[1].annotate('', (top.max(), ya), (pv_pk, ya), arrowprops=dict(arrowstyle='<->', color='#2b2a27', lw=0.9, shrinkA=0, shrinkB=0, mutation_scale=7))
ax[1].text((pv_pk + top.max())/2, ya + 50, f'lag {top.min() - pv_pk:.0f}–{top.max() - pv_pk:.0f} h', ha='center', va='bottom', fontsize=7.0)
save(f, 4)
# Fig 5 heat stratification & hedge gap
f, ax = plt.subplots(1, 3, figsize=(9.0, 3.0), gridspec_kw={'wspace': 0.42})
xb = np.arange(len(st)); RV = json.load(open('results_rev.json')); BA = RV['bin_alloc']; labs = list(st.bin)
hp = np.array([BA[b]['hp'] for b in labs]); pr = np.array([BA[b]['pr'] for b in labs])
hpe = np.array([[BA[b]['hp']-BA[b]['hp_lo'], BA[b]['hp_hi']-BA[b]['hp']] for b in labs]).T
pre = np.array([[BA[b]['pr']-BA[b]['pr_lo'], BA[b]['pr_hi']-BA[b]['pr']] for b in labs]).T
ax[0].bar(xb-0.19, hp, width=0.36, color=C['a'], yerr=hpe, capsize=2, error_kw=dict(lw=0.8), label='Heat-priority (upper bound)')
ax[0].bar(xb+0.19, pr, width=0.36, color=C['v'], yerr=pre, capsize=2, error_kw=dict(lw=0.8), label='Proportional (central)')
ax[0].set_xticks(xb, st.bin.str.replace('>=', '≥'), rotation=30); ax[0].set_ylim(0, 62); ax[0].set_ylabel('SHHR (%)'); ax[0].set_xlabel('Daily T$_{max}$ bin (°C)'); ax[0].legend(fontsize=6.5, loc='upper left', ncol=1)
ax[1].bar(xb, st.pk_red, color=C['b'], width=0.6)
for i, (v, p_) in enumerate(zip(st.pk_red, st.pk_red_pct)): ax[1].text(i, v+10, f'{p_:.0f}%', ha='center', fontsize=7)
ax[1].set_xticks(xb, st.bin.str.replace('>=', '≥'), rotation=30); ax[1].set_ylabel('Daily peak reduction (kW)'); ax[1].set_xlabel('Daily T$_{max}$ bin (°C)'); ax[1].set_title('(b) PV peak relief', loc='left')
ax[2].bar(unc.index+0.25, unc.dLh - unc.unc, width=0.45, color=C['y'], label='Covered by coincident PV (heat-priority)')
ax[2].bar(unc.index+0.25, unc.unc, bottom=unc.dLh-unc.unc, width=0.45, color=C['o'], label='Uncovered (hedge gap)')
ax[2].set_xlabel('Hour of day'); ax[2].set_ylabel('Share of temperature-driven energy (%)'); ax[2].set_xticks([0, 6, 12, 18, 24]); ax[2].legend(fontsize=6.5, loc='upper left'); ax[2].set_ylim(0, 4.8)
ax[2].set_title('(c) Timing of the hedge gap', loc='left')
save(f, 5)
# Fig 6 peak timing
f, ax = plt.subplots(1, 2, figsize=(8.4, 3.0), gridspec_kw={'wspace': 0.3})
bins_ = np.arange(9, 23, 1); cL, _ = np.histogram(pk.hL, bins_); cB, _ = np.histogram(pk.hB, bins_)
ax[0].bar(bins_[:-1]+0.5-0.2, cL, width=0.38, color=C['b'], label='Gross load peak')
ax[0].bar(bins_[:-1]+0.5+0.2, cB, width=0.38, color=C['o'], label='Net-import peak'); ax[0].set_xticks(range(9, 23, 2))
ax[0].set_xlabel('Hour of monthly on-peak maximum'); ax[0].set_ylabel('Months (n = %d)' % len(pk)); ax[0].legend(); ax[0].set_title('(a) Peak-timing shift', loc='left')
ax[0].axvspan(15, 22, color='#efeeea', zorder=0, lw=0)   # intervals ending at or after 15:00 [rev]
ax[0].text(21.8, ax[0].get_ylim()[1]*0.55, f"Peaks at or\nafter 15:00:\n{R['peaks']['share_L_after15']:.1f}% → {R['peaks']['share_B_after15']:.1f}%",
           ha='right', va='top', fontsize=7.4)
pk['m'] = pd.PeriodIndex(pk.month, freq='M').to_timestamp()
ax[1].plot(pk.m, pk.Lpk, color=C['b'], lw=1.6, marker='o', ms=3, label='Gross load'); ax[1].plot(pk.m, pk.Bpk, color=C['o'], lw=1.6, marker='o', ms=3, label='Net import')
ax[1].set_ylabel('Monthly 30-min on-peak maximum (kW)'); ax[1].legend(loc='upper left', ncol=2); ax[1].set_ylim(1200, 4200); ax[1].set_title('(b) Monthly 30-min on-peak demand proxy', loc='left')
f.autofmt_xdate(); save(f, 6)
# Fig 7 stress-test heatmaps (sequential single hue each)
ks = sorted(G.k.unique()); ks = [k for k in ks if k > 0]; dts = sorted(G.dT.unique())
f, ax = plt.subplots(1, 3, figsize=(10.6, 3.3), gridspec_kw={'wspace': 0.55})
for i, (col, lab, cmap) in enumerate([('SHHR', 'SHHR$_{hp}$ (%)', 'Blues'), ('peak_net', 'Net-peak proxy (kW)', 'Oranges'), ('CO2', 'Grid CO$_2$ (t yr$^{-1}$)', 'Greens')]):
    Z = G[G.k > 0].pivot(index='dT', columns='k', values=col).loc[dts, ks]
    im = ax[i].imshow(Z.values, origin='lower', aspect='auto', cmap=cmap)
    for a_ in range(Z.shape[0]):
        for b_ in range(Z.shape[1]):
            v = Z.values[a_, b_]; ax[i].text(b_, a_, f'{v:.0f}', ha='center', va='center', fontsize=7.2,
                                                color='white' if (v-Z.values.min())/(np.ptp(Z.values)+1e-9) > 0.6 else '#0b0b0b')
    ax[i].set_xticks(range(len(ks)), [f'{k:g}' for k in ks]); ax[i].set_yticks(range(len(dts)), [f'+{d:g}' for d in dts])
    ax[i].set_xlabel('PV scale factor k (× 2.51 MWp)'); ax[i].grid(False); ax[i].set_title(f"({'abc'[i]}) {lab}", loc='left', fontsize=8.5)
    cb = plt.colorbar(im, ax=ax[i], fraction=0.046, pad=0.03); cb.set_label(lab, fontsize=8)
ax[0].set_ylabel('Warming ΔT (°C)')
kk = [X['kstar'][d]['k_SHHR'] for d in dts]; ax[0].plot([np.interp(k, ks, range(len(ks))) for k in kk], range(len(dts)), color=C['o'], lw=2, marker='o', ms=3)
save(f, 7)
# Fig 8 battery
f, ax = plt.subplots(1, 2, figsize=(7.4, 2.8), gridspec_kw={'wspace': 0.38})
caps = sorted(Bt.cap.unique()); w = 0.2
for j, (k, c) in enumerate(zip([2, 3, 4], [C['b'], C['o'], C['a']])):
    s = Bt[(Bt.k == k) & (Bt.dT == 2.0)].set_index('cap').loc[caps]
    ax[0].bar(np.arange(len(caps))+(j-1)*w*1.1, s.curt_pct, width=w, color=c, label=f'k = {k}')
    ax[1].bar(np.arange(len(caps))+(j-1)*w*1.1, s.peak_net, width=w, color=c, label=f'k = {k}')
for a_ in ax: a_.set_xticks(range(len(caps)), [f'{c/1000:g}' for c in caps]); a_.set_xlabel('Battery capacity (MWh, 0.5C)')
ax[0].set_ylabel('PV curtailed under zero export (%)'); ax[0].legend(); ax[0].set_title('(a) Curtailment, ΔT = +2 °C', loc='left')
ax[1].axhline(G[(G.dT==0)&(G.k==1)].peak_net.iloc[0], color=C['g'], ls='--', lw=1); ax[1].text(3.4, G[(G.dT==0)&(G.k==1)].peak_net.iloc[0]+15, 'present', fontsize=7, ha='right')
ax[1].set_ylim(1800, 2600); ax[1].set_ylabel('Net-peak proxy (kW)'); ax[1].set_title('(b) Net-peak proxy, ΔT = +2 °C', loc='left')
save(f, 8)

# Graphical abstract (results-based)
RVx = json.load(open('results_extra.json'))
plt.rcParams.update({'font.size': 10.5})
f, ax = plt.subplots(1, 2, figsize=(7.0, 3.5), gridspec_kw={'wspace': 0.34, 'width_ratios': [1.25, 1]})
p = prof[prof.reg == 'hot (>=34)']
ax[0].fill_between(p.hhf, p.L-p.dLh, p.L, color=C['o'], alpha=0.25, lw=0, label='Temperature-driven load (model)')
ax[0].plot(p.hhf, p.L, color=C['b'], lw=2, label='Gross load'); ax[0].plot(p.hhf, p.B, color='#0b0b0b', lw=1.4, ls='--', label='Net import')
ax[0].plot(p.hhf, p.E, color=C['y'], lw=2, label='Rooftop PV'); ax[0].set_xticks([0, 6, 12, 18, 24]); ax[0].set_xlabel('Hour (hot working days, T$_{max}$ ≥ 34 °C)')
ax[0].set_ylabel('Mean power (kW)'); ax[0].legend(fontsize=8.6, loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=2, frameon=False)
ya = float(p.L.max())*1.06
ax[0].annotate('', xy=(15.5, ya), xytext=(13.0, ya), arrowprops=dict(arrowstyle='<->', color=C['g']))
ax[0].text(14.25, ya*1.025, '≈2.5 h lag', ha='center', fontsize=9.5, color=C['g']); ax[0].set_ylim(0, ya*1.12)
dts = [1.0, 2.0, 3.0]; ksS = [RVx['kstar'][str(d)]['k_SHHR'] for d in dts]
ksP = [RVx['kstar'][str(d)]['k_peak'] if RVx['kstar'][str(d)]['k_peak'] is not None else np.nan for d in dts]
A7 = json.load(open('results_a7.json'))['cross_model_stress'] if os.path.exists('results_a7.json') else {}
fam = [v for v in A7.values()]
def rng_(key):
    vals = [f[key] for f in fam if f.get(key) is not None and not (isinstance(f[key], float) and np.isnan(f[key]))]
    return (min(vals), max(vals)) if vals else (np.nan, np.nan)
xx = np.arange(3); top = 6.0
ax[1].bar(xx-0.19, ksS, width=0.36, color=C['a'], label='Keep present SHHR$_{hp}$')
ax[1].bar(xx+0.19, [v if not np.isnan(v) else top for v in ksP], width=0.36, color=C['o'], label='Keep present net-peak proxy')
for i, d in enumerate(dts):
    lo, hi = rng_(f'kS_hp_{d}'); ax[1].plot([i-0.19]*2, [lo, hi], color='#0b0b0b', lw=1.2)
    lo, hi = rng_(f'kP_{d}');
    if not np.isnan(lo): ax[1].plot([i+0.19]*2, [lo, min(hi, top)], color='#0b0b0b', lw=1.2)
for i, (a_, b_) in enumerate(zip(ksS, ksP)):
    ax[1].text(i-0.19, a_+0.12, f'{a_:.1f}', ha='center', fontsize=9.5)
    ax[1].text(i+0.19, (b_ if not np.isnan(b_) else top)+0.12, f'{b_:.1f}' if not np.isnan(b_) else '>6', ha='center', fontsize=9.5)
ax[1].set_xticks(xx, ['+1 °C', '+2 °C', '+3 °C']); ax[1].set_ylabel('Required PV scale factor k*'); ax[1].set_xlabel('Warming (stress test)')
ax[1].axhline(1, color=C['g'], lw=0.8, ls=':'); ax[1].set_ylim(0, 7.8); ax[1].legend(fontsize=8.6, loc='upper left')
save(f, 'GA')
print('figs done')
