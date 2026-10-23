"""Step 15 (revision, item A9): why the corrected 30-min portal peak exceeds the billed 15-min demand in some
months, and which billing-period alignment best reproduces the invoices."""
import pandas as pd, numpy as np, json, holidays
from pea_invoices import INVOICES
H = pd.read_pickle('hosp2.pkl'); iv = pd.read_pickle('iv.pkl')
inv = pd.DataFrame(INVOICES, columns=['month','kWh','kW','THB']).set_index('month')
OUT = {}
# --- 1. where and when the monthly portal maximum occurs ---
on = H[H.on & H.B.notna()].copy(); on['ms'] = on.index.strftime('%Y-%m')
rows = []
for m, g in on.groupby('ms'):
    if m not in inv.index: continue
    cov = H.loc[H.index.strftime('%Y-%m') == m, 'B'].notna().mean()
    if cov < 0.9: continue
    t = g.B.idxmax(); top = g.B.sort_values(ascending=False)*2
    day_max = (g.B*2).groupby(g.index.normalize()).max().sort_values(ascending=False)
    z = iv[(iv.index == t)].set_index('z').B*2
    rows.append(dict(month=m, portal_kW=top.iloc[0], billed_kW=inv.kW[m], ratio=top.iloc[0]/inv.kW[m],
                     t_max=str(t), day=t.day, dow=t.dayofweek, days_in_month=t.days_in_month,
                     second_day_kW=day_max.iloc[1] if len(day_max) > 1 else np.nan,
                     n_days_above_billed=int((day_max > inv.kW[m]).sum()),
                     zone1_kW=z.get(1, np.nan), zone23_kW=z.get(2, 0) + z.get(3, 0)))
P = pd.DataFrame(rows); P.round(3).to_csv('tab_peak_vs_billed.csv', index=False)
ex = P[P.ratio > 1]
OUT['exceed_months'] = ex.round(3).to_dict('records')
OUT['n_months'] = len(P); OUT['n_exceed'] = len(ex)
OUT['exceed_near_month_edge'] = int(((ex.day <= 3) | (ex.day >= ex.days_in_month - 2)).sum())
OUT['exceed_on_monday_after_weekend'] = int((ex.dow == 0).sum())
# --- 2. alternative billing-period alignments: meter read on day r of each month ---
def period_sums(r):
    idx = H.index - pd.Timedelta(days=0)
    lab = np.where(idx.day > r, (idx + pd.offsets.MonthBegin(1)).strftime('%Y-%m'), idx.strftime('%Y-%m')) if r < 28 else idx.strftime('%Y-%m')
    s = pd.DataFrame({'B': H.B.values, 'on': H.on.values, 'lab': lab}, index=H.index)
    cov = s.groupby('lab').B.count()/s.groupby('lab').size()
    e = s.groupby('lab').B.sum()/cov
    p = (s[s.on].groupby('lab').B.max()*2)
    df = pd.DataFrame({'cov': cov, 'E': e, 'P': p}).join(inv[['kWh','kW']], how='inner')
    df = df[df['cov'] > 0.9]
    return dict(read_day=r if r < 28 else 'month end', n=len(df), energy_ratio_median=float((df.E/df.kWh).median()),
                energy_ratio_IQR=float((df.E/df.kWh).quantile(.75) - (df.E/df.kWh).quantile(.25)),
                n_peak_exceed=int((df.P > df.kW).sum()), peak_ratio_median=float((df.P/df.kW).median()))
OUT['alignment'] = [period_sums(r) for r in [31, 25, 20, 15, 10, 5]]
# --- 3. holiday calendar: portal maxima on days PEA may bill as off-peak ---
th = holidays.TH(years=range(2023, 2027))
OUT['exceed_dates_holiday_names'] = {r['t_max']: th.get(pd.Timestamp(r['t_max']).date()) for r in OUT['exceed_months']}
json.dump(OUT, open('results_a15.json', 'w'), indent=1, default=str)
print(ex[['month','portal_kW','billed_kW','ratio','t_max','second_day_kW','n_days_above_billed','zone1_kW','zone23_kW']].round(2).to_string())
print('edge', OUT['exceed_near_month_edge'], 'monday', OUT['exceed_on_monday_after_weekend'])
print(pd.DataFrame(OUT['alignment']).round(4).to_string())

# --- 4. counter-timing artefacts: a low interval followed by a high one whose pair mean matches the neighbours ---
def pairs(s):
    loc = (s.shift(2) + s.shift(-1))/2
    return (s.shift(1)/loc < 0.8) & (s/loc > 1.2) & ((s.shift(1) + s)/(2*loc)).between(0.9, 1.1)
def smooth(s):
    f = pairs(s); m = (s.shift(1) + s)/2
    s2 = s.copy(); s2[f] = m[f]; s2[f.shift(-1, fill_value=False)] = m.shift(-1)[f.shift(-1, fill_value=False)]
    return s2
Hs = H.copy(); Hs['B'] = smooth(H.B); Hs['L'] = smooth(H.L)
OUT['artefact_pairs'] = dict(B=int(pairs(H.B).sum()), L=int(pairs(H.L).sum()))
pk = pd.read_csv('peaks_monthly.csv'); months = set(pk.month)
def pkstats(D):
    o = D[D.on & D.B.notna() & D.L.notna()]; o = o[o.month.astype(str).isin(months)]; g = o.groupby(o.month.astype(str))
    hB = g.B.idxmax().map(lambda t: t.hour + t.minute/60)
    return dict(gross=float((g.L.max()*2).mean()), net=float((g.B.max()*2).mean()), relief=float(((g.L.max() - g.B.max())*2).mean()),
                net_after15=float((hB >= 15).mean()*100))
OUT['peaks_raw'] = pkstats(H); OUT['peaks_pair_smoothed'] = pkstats(Hs)
ons = Hs[Hs.on & Hs.B.notna()]; mx = ons.groupby(ons.index.strftime('%Y-%m')).B.max()*2
Q = P.set_index('month').join(mx.rename('smoothed_kW'))
OUT['n_exceed_after_smoothing'] = int((Q.smoothed_kW > Q.billed_kW).sum())
OUT['exceed_after_smoothing'] = (Q[Q.smoothed_kW > Q.billed_kW].smoothed_kW/Q[Q.smoothed_kW > Q.billed_kW].billed_kW).round(3).to_dict()
json.dump(OUT, open('results_a15.json', 'w'), indent=1, default=str)
print('pairs', OUT['artefact_pairs']); print('raw', OUT['peaks_raw']); print('smoothed', OUT['peaks_pair_smoothed'])
print('exceed after smoothing', OUT['n_exceed_after_smoothing'], OUT['exceed_after_smoothing'])
