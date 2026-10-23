"""Step 12: preservation multipliers for the raw (uncalibrated) portal import, as a sensitivity case."""
import json, numpy as np, pandas as pd
from pathlib import Path
src = (Path(__file__).resolve().parent / 'a8_finegrid.py').read_text(encoding='utf-8').split('OUT = {}')[0]
c = float(open('c_ct.txt').read())
src = src.replace("S = H.dropna(", "H['L'] = (H.E.fillna(0) + H.B*%r - H.X.fillna(0)*%r).clip(lower=0).where(H.L.notna()); M['L'] = (M.E.fillna(0) + M.B*%r - M.X.fillna(0)*%r).clip(lower=0)\nS = H.dropna(" % (c, c, c, c), 1)
exec(src)
ref = FAM['GBM, monotone (reference)'](TN0).fit(M); b0 = stress(ref, TN0, 0.0).iloc[0]; out = {}
for dT in [1.0, 2.0, 3.0]:
    df = stress(ref, TN0, dT)
    out[dT] = dict(kS_hp=kstar(b0.SHHR_hp, df, 'SHHR_hp'), kS_pr=kstar(b0.SHHR_pr, df, 'SHHR_pr'), kP=kstar(b0.peak, df, 'peak', decreasing=True))
out['base'] = dict(SHHR_hp=b0.SHHR_hp, SHHR_pr=b0.SHHR_pr, peak=b0.peak, c=c)
print(out); json.dump(out, open('results_a12.json', 'w'), indent=1, default=float)
