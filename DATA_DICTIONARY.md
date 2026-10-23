# Data dictionary: data/hospital_30min_series.csv

61,680 rows, one row per 30-min interval on the full grid (written by `a20_export_dataset.py`). Interval energies are hospital totals of the three PV zones and are empty where any zone failed QC.

| Column | Description | Unit | Notes |
|---|---|---|---|
| interval_end_local_time | End of the 30-min interval | local time (UTC+7) | 2023-03-03 00:00 to 2026-09-07 23:30 |
| pv_kWh | PV energy delivered, E_t | kWh | Difference of lifetime counters |
| gross_load_kWh | Gross electrical load, L_t | kWh | E + B − X with corrected B and X |
| grid_import_kWh | Grid import, B_t, corrected for the CT-ratio error | kWh | Portal value / 1.20, counter-timing pairs replaced by their mean |
| grid_import_portal_raw_kWh | Grid import as recorded by the portal | kWh | Uncorrected (sensitivity) |
| export_kWh | Exported energy, X_t, corrected for the CT-ratio error | kWh | Portal value / 1.20 |
| ambient_temp_C | Ambient temperature | °C | Portal weather sensor |
| ghi_W_m2 | Global horizontal irradiance | W/m² | Portal weather sensor |
| non_working_day | Weekend or Thai public holiday | bool | `holidays` library |
