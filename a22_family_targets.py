"""Step 22 (revision): absolute energy and peak preservation multipliers (k*imp, k*peak) and their ratio for the six
model families at every warming level of the stress test (results_a22.json, used by Fig. 6). Same procedure as the
+2 degC family comparison in a13_framing.py."""
import pandas as pd, numpy as np, json, warnings
from pathlib import Path
warnings.filterwarnings('ignore')
src = (Path(__file__).resolve().parent / 'a8_finegrid.py').read_text(encoding='utf-8').split('OUT = {}')[0]
exec(src)                                    # a7 helpers, model families, S, KS, Tc0, GAM, kstar
on = S.on.values.astype(bool); mon = S.month.astype(str).values
DTS = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
def targets(m):
    g0 = m.predict(S); L0 = S.L.values
    gross0 = pd.Series(L0[on]*2).groupby(mon[on]).max()
    imp0 = (L0 - np.minimum(S.E.values, L0)).sum(); pk0 = pd.Series((L0 - np.minimum(S.E.values, L0))[on]*2).groupby(mon[on]).max().mean()
    out = {}
    for dT in DTS:
        X = S.copy(); X[TFE] = X[TFE] + dT
        L = np.clip(L0 + m.predict(X) - g0, 0, None); fac = (1 + GAM*(Tc0 + dT - 25))/(1 + GAM*(Tc0 - 25)); rows = []
        for k in KS:
            E = S.E.values*k*fac; imp = L - np.minimum(E, L)
            rows.append(dict(k=k, imp=imp.sum(), peak=pd.Series(imp[on]*2).groupby(mon[on]).max().mean()))
        df = pd.DataFrame(rows)
        ki, kp = kstar(imp0, df, 'imp', decreasing=True), kstar(pk0, df, 'peak', decreasing=True)
        out[f'{dT:.1f}'] = dict(k_import=ki, k_peak=kp, ratio=(kp/ki if ki == ki and kp == kp else float('nan')))
    return out
OUT = {}
for name, mkf in FAM.items():
    OUT[name] = targets(mkf(TN0).fit(M)); print(name, {d: (round(r['k_import'], 2), round(r['k_peak'], 2)) for d, r in OUT[name].items()})
json.dump(OUT, open('results_a22.json', 'w'), indent=1, default=float)
