"""Step 20 (revision): write the public hospital-level dataset data/hospital_30min_series.csv from the pipeline itself
(the earlier file was produced outside the package). One row per 30-min interval on the full grid; interval energies
are empty where any zone failed quality control."""
import pandas as pd, numpy as np, os
H = pd.read_pickle('hosp.pkl'); Hr = pd.read_pickle('hosp_raw.pkl')
out = pd.DataFrame(index=H.index)
out.index.name = 'interval_end_local_time'
out['pv_kWh'] = H.E
out['gross_load_kWh'] = H.L                       # E + B - X, from CT-corrected and timing-checked import and export
out['grid_import_kWh'] = H.B                      # portal import / 1.20, counter-timing pairs replaced by their mean
out['grid_import_portal_raw_kWh'] = Hr.B.reindex(H.index)   # as recorded by the portal
out['export_kWh'] = H.X
out['ambient_temp_C'] = H.Tamb
out['ghi_W_m2'] = H.G
out['non_working_day'] = H.hol.astype(bool)
os.makedirs('data', exist_ok=True)
out.round(4).to_csv('data/hospital_30min_series.csv')
print('rows', len(out), out.index.min(), out.index.max(), 'valid load rows', int(out.gross_load_kWh.notna().sum()))
