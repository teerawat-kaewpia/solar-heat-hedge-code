"""Step 1: reconstruct 30-min interval energy from cumulative counters + QC."""
import pandas as pd, numpy as np, holidays, sys, os
# usage: python a1_prep.py [path/to/CRH_merged_30min_zone_level.csv]
RAW = sys.argv[1] if len(sys.argv) > 1 else os.environ.get('CRH_DATA', 'data/CRH_merged_30min_zone_level.csv')
cols = ['src','plant','time','kWp','type','built','grid','status','E_day','E_tot','L_day','L_tot',
        'B_day','B_tot','X_day','X_tot','h_eq','PR','G','H_day','curt','Tamb','P','Lreal']
df = pd.read_csv(RAW, encoding='utf-8-sig'); df.columns = cols
df['time'] = pd.to_datetime(df.time, format='%Y-%m-%d %H:%M' if df.time.astype(str).str.match(r'\d{4}-').all() else '%m/%d/%Y %H:%M')   # portal export (ISO) or Excel-resaved copy [rev]
df['z'] = df.plant.str[-1].astype(int)
df['st'] = df.status.map({'ปกติ':'normal','สัญญาณเตือน':'alarm','ฟอลต์':'fault'})
out, qc = [], []
for z, d in df.groupby('z'):
    d = d.sort_values('time').drop_duplicates('time', keep='last').set_index('time')
    idx = pd.date_range(d.index.min(), d.index.max(), freq='30min')
    n_dup = df[df.z==z].shape[0]-len(d)
    d = d.reindex(idx)
    kwp = d.kWp.dropna().iloc[0]
    r = pd.DataFrame(index=idx); r['z'] = z; r['kWp'] = kwp
    for c in ['E','L','B','X']:
        v = d[f'{c}_tot'].diff()
        r[c] = v
    # physical QC
    bad = (r[['E','L','B','X']] < -1e-6).any(axis=1)
    bad |= r.E > 0.5*kwp*1.05
    for c in ['L','B']:
        q = r[c].quantile(0.9995); bad |= r[c] > 1.5*q
    bad |= d.E_tot.isna() | d.E_tot.shift().isna()
    r.loc[bad, ['E','L','B','X']] = np.nan
    # Hampel-type spike/dropout filter on load (ratio to centred 7-day rolling median)
    rm = r.L.rolling(48*7, center=True, min_periods=48).median()
    ratio = r.L / rm
    spk = (ratio > 2.2) | (ratio < 0.35)
    r.loc[spk, ['E','L','B','X']] = np.nan
    bad |= spk.fillna(False)
    r['G'] = d.G.clip(0, 1500); r['Tamb'] = d.Tamb; r['P'] = d.P; r['st'] = d.st
    r.loc[(r.index.hour < 5) | (r.index.hour >= 20), 'G'] = r.loc[(r.index.hour < 5) | (r.index.hour >= 20), 'G'].fillna(0)
    # energy balance closure check L = E + B - X
    res = (r.L - (r.E + r.B - r.X)).abs()
    qc.append(dict(zone=z, kWp=kwp, n_raw=int((df.z==z).sum()), n_dup=int(n_dup), n_expected=len(idx),
                   n_missing=int(d.E_tot.isna().sum()), n_flag=int(bad.sum()),
                   valid_pct=round(100*r.L.notna().mean(),2), closure_med=round(res.median(),4),
                   closure_p99=round(res.quantile(.99),3)))
    out.append(r)
iv = pd.concat(out); iv.index.name = 'time'
iv.to_pickle('iv_raw.pkl')
pd.DataFrame(qc).to_csv('tab_qc.csv', index=False)
print(pd.DataFrame(qc).to_string())
# hospital aggregate (all three zones valid)
w = iv.pivot_table(index='time', columns='z', values=['E','L','B','X','G','Tamb'])
w = w.reindex(pd.date_range(w.index.min(), w.index.max(), freq='30min'))   # pivot_table drops all-NaN rows; restore the full 30-min grid [rev A3]
H = pd.DataFrame(index=w.index)
for c in ['E','L','B','X']:
    H[c] = w[c].sum(axis=1, min_count=3)
H['Tamb'] = w['Tamb'].median(axis=1); H['G'] = w['G'].median(axis=1)
H['kWp'] = iv.groupby('z').kWp.first().sum()
th = holidays.TH(years=range(2023, 2027))
H['date'] = H.index.normalize()
H['hol'] = H.date.dt.date.map(lambda x: x in th) | (H.index.dayofweek >= 5)
H.to_pickle('hosp_raw.pkl')
# ---- calibration of grid import to the utility revenue meter (PEA invoices) ----
from pea_invoices import INVOICES
inv = pd.DataFrame(INVOICES, columns=['month', 'kWh', 'kW', 'THB']).set_index('month')
m = H.index.strftime('%Y-%m'); g = H.B.groupby(m); cov = g.count()/(pd.PeriodIndex(g.count().index, freq='M').days_in_month*48)
Bm = (g.sum()/cov)[cov > 0.9]; Bm = Bm[Bm.index.isin(inv.index)]
ratio = Bm/inv.kWh.reindex(Bm.index)
C_INV = float(ratio.median())
pd.DataFrame({'portal_B_kWh': Bm, 'invoice_kWh': inv.kWh.reindex(Bm.index), 'ratio': ratio, 'coverage': cov.reindex(Bm.index)}).to_csv('tab_invoice_ratio.csv')
print('invoice calibration: n=%d months, median=%.4f, IQR=%.4f-%.4f, SD=%.4f' % (len(ratio), C_INV, ratio.quantile(.25), ratio.quantile(.75), ratio.std()))
open('c_inv.txt', 'w').write(repr(C_INV))
# Correction factor: current-transformer (CT) ratio error confirmed on site in all three portal meters
# (configured ratio = 1.20 x installed ratio). The invoice median C_INV is used only as an independent check.
C_CT = 1.20
open('c_ct.txt', 'w').write(repr(C_CT))
print('CT correction 1.20; invoice ratio after correction: median %.4f' % (ratio/C_CT).median())
# grid import and export (same CT) corrected in every zone; gross load rebuilt from the energy balance
iv['B'] = iv.B/C_CT; iv['X'] = iv.X/C_CT; iv['L'] = iv.E + iv.B - iv.X
for c in ['B', 'L']: H[c] = H[c]/1.0
H['B'] = H.B/C_CT; H['X'] = H.X/C_CT; H['L'] = H.E + H.B - H.X
# counter-timing artefacts: a low interval followed by a high one whose mean matches the neighbours
# (energy logged one interval late). Replace both by their mean (energy-preserving) and rebuild L. [rev A9]
loc = (H.B.shift(2) + H.B.shift(-1))/2
pair = (H.B.shift(1)/loc < 0.8) & (H.B/loc > 1.2) & ((H.B.shift(1) + H.B)/(2*loc)).between(0.9, 1.1)
mean2 = (H.B.shift(1) + H.B)/2; first = pair.shift(-1, fill_value=False)
H.loc[pair, 'B'] = mean2[pair]; H.loc[first, 'B'] = mean2.shift(-1)[first]
H['L'] = H.E + H.B - H.X
open('qc_timing_pairs.txt', 'w').write(str(int(pair.sum())))
print('counter-timing pairs smoothed in grid import:', int(pair.sum()))
iv.to_pickle('iv.pkl'); H.to_pickle('hosp.pkl')
print(H.describe().T.round(2).to_string())
print('hospital valid %', H.L.notna().mean()*100)
