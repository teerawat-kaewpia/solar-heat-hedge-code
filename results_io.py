"""Exact, pickle-free storage of the two intermediate tables (hosp2, daily) that the figure scripts and the value
definitions read. Each table is a gzip CSV plus a JSON schema of column types, and the round trip is exact.
export: python results_io.py export <run_dir> <results_dir>   (used when the package is built)
The reviewer entry point reproduce.py calls restore() to rebuild hosp2.pkl and daily.pkl in its working folder."""
import json, sys, os
import pandas as pd

TABLES = ['hosp2', 'daily']

def _schema(df):
    cols = {}
    for c, t in df.dtypes.items():
        if isinstance(t, pd.CategoricalDtype):
            cols[c] = dict(kind='category', categories=[str(x) for x in t.categories], ordered=bool(t.ordered))
        elif isinstance(t, pd.PeriodDtype):
            cols[c] = dict(kind='period', freq=str(t)[len('period['):-1])   # 'period[M]' -> 'M'
        else:
            cols[c] = dict(kind='plain', dtype=str(t))
    return dict(index_name=df.index.name, index_dtype=str(df.index.dtype), columns=cols, order=list(df.columns))

def export(run_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for name in TABLES:
        df = pd.read_pickle(os.path.join(run_dir, name + '.pkl'))
        df.to_csv(os.path.join(out_dir, name + '.csv.gz'), compression={'method': 'gzip', 'mtime': 0})
        json.dump(_schema(df), open(os.path.join(out_dir, name + '.schema.json'), 'w', encoding='utf-8'), indent=1)
        back = load(out_dir, name)
        pd.testing.assert_frame_equal(df, back, check_exact=True, check_freq=False)   # refuse to ship an inexact copy
        print(f'{name}: {df.shape}, round trip exact')

def load(res_dir, name):
    sc = json.load(open(os.path.join(res_dir, name + '.schema.json'), encoding='utf-8'))
    df = pd.read_csv(os.path.join(res_dir, name + '.csv.gz'), index_col=0, float_precision='round_trip', keep_default_na=True)
    df.index = pd.to_datetime(df.index).astype(sc['index_dtype']); df.index.name = sc['index_name']
    for c, spec in sc['columns'].items():
        if spec['kind'] == 'category':
            df[c] = pd.Categorical(df[c], categories=spec['categories'], ordered=spec['ordered'])
        elif spec['kind'] == 'period':
            df[c] = pd.PeriodIndex(df[c], freq=spec['freq'])
        elif spec['dtype'].startswith('datetime64'):
            df[c] = pd.to_datetime(df[c]).astype(spec['dtype'])
        elif spec['dtype'] == 'bool':
            df[c] = df[c].astype(str).map({'True': True, 'False': False}).astype(bool)
        else:
            df[c] = df[c].astype(spec['dtype'])
    return df[sc['order']]

def restore(res_dir, work_dir):
    for name in TABLES:
        load(res_dir, name).to_pickle(os.path.join(work_dir, name + '.pkl'))

if __name__ == '__main__' and sys.argv[1:2] == ['export']:
    export(sys.argv[2], sys.argv[3])
