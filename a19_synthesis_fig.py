"""Step 19 (revision): synthesis figure (Fig. S5). Every number in the boxes is read from the result files,
replacing the hard-coded text formerly in a6_scenarios.py."""
import json, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
J = lambda f: json.load(open(f, encoding='utf-8'))
R, RV, VAL, A13, A8 = J('results.json'), J('results_rev.json'), J('results_val.json'), J('results_a13.json'), J('results_a8.json')
A10 = J('results_a10.json'); Bt = pd.read_csv('battery.csv'); H = pd.read_pickle('hosp2.pkl').dropna(subset=['E', 'dLh'])
hh = H.index.hour + H.index.minute/60; pD = H.groupby(hh).dLh.mean(); pE = H.groupby(hh).E.mean()
top = pD[pD >= 0.95*pD.max()].index; lag = (top.min() - pE.idxmax(), top.max() - pE.idxmax())
ip = RV['ind_point']; sh = [s['heat_share'] for s in VAL['spec'].values()]; fam = A13['families_2C']
kimp = [v['k_import'] for v in fam.values()]; kpk = [v['k_peak'] for v in fam.values()]
fine2 = [r for r in A8['fine'] if r['dT'] == 2.0][0]; pk = R['peaks']; ec = R['econ']
b = Bt[(Bt.dT == 2.0) & (Bt.k == 3) & (Bt.cap == 5000)].iloc[0]; present = pk['Bpk_mean']
C = dict(b='#2a78d6', o='#eb6834', a='#1baf7a', y='#eda100', v='#4a3aa7', g='#52514e', m='#e87ba4')
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8})
f, ax = plt.subplots(figsize=(9.2, 4.6)); ax.axis('off'); ax.set_xlim(0, 12); ax.set_ylim(0, 6)
def box(x, y, w, h, t, c, fs=7.4):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.04,rounding_size=0.12', fc=c+'22', ec=c, lw=1.4))
    ax.text(x+w/2, y+h/2, t, ha='center', va='center', fontsize=fs)
def arr(x0, y0, x1, y1): ax.annotate('', (x1, y1), (x0, y0), arrowprops=dict(arrowstyle='-|>', color=C['g'], lw=1.2))
box(0.1, 4.2, 2.1, 1.5, f"Ambient temperature\n(T, 24-h, 72-h means)\nT$_b$ ≈ {R['cp_hospital']['Tb']:.1f} °C", C['b'])
box(2.6, 4.2, 2.3, 1.5, f"Model-attributed\ntemperature-driven load\n≈ {ip['heat_share']:.0f}% of demand\n({min(sh):.0f}–{max(sh):.0f}%, 6 models)", C['o'])
box(5.3, 4.2, 2.3, 1.5, f"Thermal lag\nbroad heat-load maximum\n{lag[0]:.0f}–{lag[1]:.0f} h after PV peak\n{RV['night_share']:.0f}% at night", C['v'])
box(8.0, 4.2, 3.9, 1.5, (f"Solar-heat hedge (diagnostic)\nHEOR {ip['HEOR']:.0f}%  |  SHHR ≈ {ip['SHHR_pr']:.0f}–{A10['calibrated']['primary']['SHHR_sh']:.0f}%\n"
                         f"(neutral rules, bounds {ip['SHHR_bl']:.0f}–{ip['SHHR_hp']:.0f}%)\nη$_{{hedge}}$ = {RV['elastic']['eta']:.2f}"), C['a'])
box(8.0, 2.2, 3.9, 1.5, (f"Peak displacement\n30-min on-peak proxy −{pk['dP_mean']:.0f} kW\npeaks ≥ 15:00: {pk['share_L_after15']:.0f}% → {pk['share_B_after15']:.0f}%\n"
                         f"demand cost ≈ {ec['demand_share_saving']:.0f}% of\ntariff-equivalent saving"), C['y'])
box(4.3, 2.2, 3.3, 1.5, 'Warming stress test\nΔT = +2 °C (within the IPCC AR6\nlate-century warming range)', C['m'])
box(0.1, 0.2, 5.6, 1.5, (f"PV-only preservation multipliers at +2 °C (site-specific)\nshares: k*$_{{SHHR}}$ ≈ {fine2['kS_hp']:.2f}×, net/gross peak ≈ {A13['like_for_like'][2]['k_peak_ratio']:.2f}×\n"
                         f"absolute: grid import {min(kimp):.1f}–{max(kimp):.1f}×  vs  net-peak proxy {min(kpk):.1f}–{max(kpk):.1f}×\n(six model families)"), C['o'])
box(6.3, 0.2, 5.6, 1.5, (f"Complementary late-afternoon flexibility\nstorage, chilled-water/ice storage, pre-cooling,\nwest-facing PV (screening at +2 °C:\n"
                         f"k = 3 + 5 MWh → net-peak proxy {(1 - b.peak_net/present)*100:.0f}% below present)"), C['a'])
arr(2.2, 4.95, 2.6, 4.95); arr(4.9, 4.95, 5.3, 4.95); arr(7.6, 4.95, 8.0, 4.95); arr(9.95, 4.2, 9.95, 3.7)
arr(8.0, 2.95, 7.6, 2.95); arr(5.95, 2.2, 3.2, 1.7); arr(5.95, 2.2, 8.9, 1.7)
f.savefig('fig10.png', bbox_inches='tight', dpi=900); plt.close(f); print('fig S5 (synthesis) done')
