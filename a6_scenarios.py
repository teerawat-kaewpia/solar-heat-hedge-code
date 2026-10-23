"""Step 6: (i) map IPCC AR6 (2023) warming levels onto the stress-test axis,
(ii) conceptual synthesis figure (Fig. 9)."""
import json, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
G = pd.read_csv('stress_grid.csv'); X = json.load(open('results_extra.json'))
# IPCC AR6 Synthesis Report (2023) best-estimate global warming vs 1850-1900; 2011-2020 = 1.1 degC
ipcc = [('SSP2-4.5', '2041–2060', 2.0), ('SSP5-8.5', '2041–2060', 2.4), ('SSP2-4.5', '2081–2100', 2.7), ('SSP5-8.5', '2081–2100', 4.4)]
dts = sorted(G.dT.unique()); kS = [X['kstar'][str(float(d))]['k_SHHR'] for d in dts]
kP = [X['kstar'][str(float(d))]['k_peak'] if X['kstar'][str(float(d))]['k_peak'] is not None else np.nan for d in dts]
g1 = G[G.k == 1].set_index('dT'); rows = []
for ssp, per, glob in ipcc:
    d = round(glob - 1.1, 2); inr = d <= max(dts)
    rows.append(dict(scenario=ssp, period=per, global_warming=glob, dT_vs_2011_2020=d,
                     L=np.interp(d, dts, g1.L_GWh.loc[dts]) if inr else np.nan, SHHR_hp=np.interp(d, dts, g1.SHHR.loc[dts]) if inr else np.nan,
                     peak=np.interp(d, dts, g1.peak_net.loc[dts]) if inr else np.nan, kS=np.interp(d, dts, kS) if inr else np.nan,
                     kP=np.interp(d, dts, kP) if d <= 2.5 else np.nan))
T = pd.DataFrame(rows); T.to_csv('tab_ipcc_envelope.csv', index=False); print(T.round(2).to_string())
json.dump(T.to_dict('records'), open('results_ipcc.json', 'w'), indent=1, default=float)

# synthesis figure moved to a19_synthesis_fig.py, which reads every number from the result files [rev]
