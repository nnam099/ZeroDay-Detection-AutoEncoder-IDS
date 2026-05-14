"""UNSW-NB15 loading, feature engineering, RobustScaler fitting and weighted DataLoader creation."""

import os, glob, random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.model_selection import train_test_split

KNOWN_ATTACK_CATS  = ['Normal','DoS','Exploits','Reconnaissance','Generic']
ZERO_DAY_ATTACK_CATS = ['Fuzzers','Analysis','Backdoors','Shellcode','Worms']

UNSW_RAW_COLUMNS = [
    'srcip','sport','dstip','dsport','proto',
    'state','dur','sbytes','dbytes','sttl',
    'dttl','sloss','dloss','service','sload',
    'dload','spkts','dpkts','swin','dwin',
    'stcpb','dtcpb','smeansz','dmeansz','trans_depth',
    'res_bdy_len','sjit','djit','stime','ltime',
    'sintpkt','dintpkt','tcprtt','synack','ackdat',
    'is_sm_ips_ports','ct_state_ttl','ct_flw_http_mthd','is_ftp_login','ct_ftp_cmd',
    'ct_srv_src','ct_srv_dst','ct_dst_ltm','ct_src_ltm','ct_src_dport_ltm',
    'ct_dst_sport_ltm','ct_dst_src_ltm','attack_cat','label',
]

SKIP_FILES = {
    'NUSW-NB15_features.csv','UNSW-NB15_features.csv',
    'UNSW-NB15_LIST_EVENTS.csv','UNSW-NB15_GT.csv',
}

# ═══════════════════════════════════════════════════════════════
# DATA PIPELINE
# ═══════════════════════════════════════════════════════════════
def _find_unsw_csvs(data_dir):
    all_csvs = sorted(glob.glob(os.path.join(data_dir,'**/*.csv'), recursive=True))
    all_csvs += sorted(glob.glob(os.path.join(data_dir,'*.csv')))
    all_csvs = list(dict.fromkeys(all_csvs))
    out = []
    for p in all_csvs:
        name = os.path.basename(p)
        if name in SKIP_FILES:
            print(f'  [SKIP] {name}')
            continue
        if 'UNSW' in name.upper() or 'NB15' in name.upper():
            out.append(p)
    return out


def load_unsw_csvs(data_dir):
    if not os.path.exists(data_dir) and os.path.exists('/kaggle/input'):
        data_dir = '/kaggle/input'
    csv_files = _find_unsw_csvs(data_dir)
    if not csv_files:
        raise FileNotFoundError(f'No UNSW-NB15 CSV in {data_dir}')
    dfs = []
    for path in csv_files:
        name = os.path.basename(path)
        try:
            probe = pd.read_csv(path, nrows=3, low_memory=False,
                                encoding='utf-8', on_bad_lines='skip')
            has_hdr = any('attack_cat' in str(c).lower() for c in probe.columns)
            if has_hdr:
                df = pd.read_csv(path, low_memory=False,
                                 encoding='utf-8', on_bad_lines='skip')
                df.columns = [str(c).strip().lower().replace(' ','_') for c in df.columns]
            else:
                nc = probe.shape[1]
                cols = UNSW_RAW_COLUMNS if nc == 49 else \
                       [c for c in UNSW_RAW_COLUMNS if c not in ('stime','ltime')] if nc == 47 \
                       else [f'col_{i}' for i in range(nc)]
                df = pd.read_csv(path, header=None, names=cols,
                                 low_memory=False, encoding='latin-1', on_bad_lines='skip')
                df.columns = [str(c).strip().lower() for c in df.columns]
            print(f'  [OK] {name:45s} {len(df):>8,} rows')
            dfs.append(df)
        except Exception as e:
            print(f'  [ERR] {name}: {e}')
    df = pd.concat(dfs, ignore_index=True)
    print(f'\n  Total: {len(df):,} rows')
    return df


def normalize_labels(df):
    ac_col = next((c for c in df.columns if 'attack_cat' in c.lower()), None)
    lb_col = next((c for c in df.columns if c.lower() == 'label'), None)
    if ac_col is None:
        ac_col = df.columns[-2]
    df['attack_cat'] = df[ac_col].astype(str).str.strip()
    df['attack_cat'] = df['attack_cat'].replace(['nan','NaN','',' ','None','-'], 'Normal')
    cat_map = {
        'normal':'Normal','dos':'DoS','exploits':'Exploits','exploit':'Exploits',
        'reconnaissance':'Reconnaissance','generic':'Generic','fuzzers':'Fuzzers',
        'fuzzer':'Fuzzers','analysis':'Analysis','backdoor':'Backdoors',
        'backdoors':'Backdoors','shellcode':'Shellcode','worms':'Worms','worm':'Worms',
    }
    df['attack_cat'] = df['attack_cat'].str.lower().map(
        lambda x: cat_map.get(x.strip(), x.capitalize()))
    if lb_col and lb_col != ac_col:
        df['label_binary'] = pd.to_numeric(df[lb_col], errors='coerce').fillna(0).astype(int)
    else:
        df['label_binary'] = (df['attack_cat'] != 'Normal').astype(int)
    return df


def _encode_categorical_features(df, categorical_maps=None):
    categorical_maps = categorical_maps or {}
    fitted_maps = {}
    for cat in ['proto','service','state']:
        if cat not in df.columns:
            continue
        values = df[cat].astype(str).fillna('unk')
        if cat in categorical_maps:
            mapping = categorical_maps[cat]
        else:
            classes = sorted(values.unique().tolist())
            if 'unk' not in classes:
                classes.append('unk')
            mapping = {v: i for i, v in enumerate(classes)}
        df[f'{cat}_num'] = values.map(lambda x: mapping.get(x, mapping.get('unk', -1))).astype(np.float32)
        fitted_maps[cat] = mapping
    return fitted_maps


def _get_numeric_features(df, categorical_maps=None):
    exclude = {'attack_cat','label','label_binary','srcip','dstip',
               'sport','dsport','stime','ltime','id','proto','service','state'}
    fitted_maps = _encode_categorical_features(df, categorical_maps)
    cols = []
    for c in df.columns:
        if c in exclude: continue
        if c.endswith('_num'):
            cols.append(c); continue
        try:
            if np.issubdtype(df[c].dtype, np.number):
                cols.append(c)
        except: pass
    return cols, fitted_maps


def engineer_features(df, feat_cols):
    eps = 1e-8
    new = []
    def add(name, vals):
        arr = np.asarray(vals, dtype=np.float32)
        arr = np.where(np.isfinite(arr), arr, 0.)
        df[name] = arr; new.append(name)

    if 'sbytes' in df.columns and 'dbytes' in df.columns:
        tb = df['sbytes'] + df['dbytes'] + eps
        add('bytes_ratio', df['sbytes']/tb)
        add('log_total_bytes', np.log1p(tb))
        add('log_sbytes', np.log1p(df['sbytes'].clip(lower=0)))
        add('log_dbytes', np.log1p(df['dbytes'].clip(lower=0)))

    if 'spkts' in df.columns and 'dpkts' in df.columns:
        tp = df['spkts'] + df['dpkts'] + eps
        add('pkts_ratio', df['spkts']/tp)
        add('log_total_pkts', np.log1p(tp))

    if 'sbytes' in df.columns and 'dur' in df.columns:
        dur_s = df['dur'].clip(lower=1e-6)
        add('src_bps',     df['sbytes']/dur_s)
        add('log_src_bps', np.log1p(df['sbytes']/dur_s))
        add('pkt_rate',    (df.get('spkts', pd.Series(0,index=df.index))+eps)/dur_s)

    if 'sload' in df.columns and 'dload' in df.columns:
        tl = df['sload'] + df['dload'] + eps
        add('load_asym', (df['sload']-df['dload']).abs()/tl)
        add('log_sload', np.log1p(df['sload'].clip(lower=0)))

    if 'sttl' in df.columns and 'dttl' in df.columns:
        add('ttl_diff', (df['sttl']-df['dttl']).abs())
        add('ttl_sum',  df['sttl']+df['dttl'])

    if 'sloss' in df.columns and 'spkts' in df.columns:
        add('loss_rate_src', df['sloss']/(df['spkts']+eps))
    if 'dloss' in df.columns and 'dpkts' in df.columns:
        add('loss_rate_dst', df['dloss']/(df['dpkts']+eps))

    if 'sjit' in df.columns and 'djit' in df.columns:
        add('jit_ratio', df['sjit']/(df['djit']+eps))
        add('log_sjit',  np.log1p(df['sjit'].clip(lower=0)))

    if 'synack' in df.columns and 'ackdat' in df.columns:
        add('handshake_ratio', df['synack']/(df['ackdat']+eps))
        add('incomplete_tcp',  ((df['synack']>0)&(df['ackdat']==0)).astype(float))

    if 'sintpkt' in df.columns and 'dintpkt' in df.columns:
        add('intpkt_ratio', df['sintpkt']/(df['dintpkt']+eps))

    return df, feat_cols + new


def clean_df(df):
    df = df.replace([np.inf,-np.inf], np.nan)
    n = len(df)
    df = df.drop_duplicates()
    print(f'  Removed {n-len(df):,} duplicates')
    return df


def prepare_splits(df, known_cats=KNOWN_ATTACK_CATS, zd_cats=ZERO_DAY_ATTACK_CATS,
                   test_ratio=0.20, val_ratio=0.10, seed=42, zd_augment=1):
    print('\n[DATA SPLIT]')
    df = normalize_labels(df)

    cat_counts = df['attack_cat'].value_counts()
    print('\n  Distribution:')
    for cat, cnt in cat_counts.items():
        m = 'K' if cat in known_cats else ('Z' if cat in zd_cats else '?')
        print(f'    [{m}] {cat:<22} {cnt:>8,}')

    avail      = set(df['attack_cat'].unique())
    act_known  = [c for c in known_cats if c in avail]
    act_zd     = [c for c in zd_cats    if c in avail]

    for cat in avail - set(act_known) - set(act_zd):
        n = (df['attack_cat']==cat).sum()
        if n < 5000: act_zd.append(cat)
        else: act_known.append(cat)

    df_known   = df[df['attack_cat'].isin(act_known)].copy()
    df_zd_full = df[df['attack_cat'].isin(act_zd)].copy()

    print(f'\n  Known  {len(act_known)} classes: {len(df_known):,} samples')
    print(f'  ZD     {len(act_zd)} classes: {len(df_zd_full):,} samples')

    feat_cols, categorical_maps = _get_numeric_features(df_known)
    df_known, feat_cols = engineer_features(df_known, feat_cols)
    zd_feat_cols, _     = _get_numeric_features(df_zd_full, categorical_maps)
    df_zd_full, _       = engineer_features(df_zd_full, zd_feat_cols)
    feat_cols = [c for c in feat_cols if c in df_zd_full.columns]

    std = df_known[feat_cols].std()
    feat_cols = [c for c in feat_cols if std[c] > 1e-8]
    print(f'  Features: {len(feat_cols)}')

    le = LabelEncoder()
    le.fit(act_known)
    df_known['y'] = le.transform(df_known['attack_cat'])
    n_classes = len(act_known)

    min_s = df_known['y'].value_counts().min()
    strat = df_known['y'] if min_s >= 5 else None

    idx_tv, idx_te = train_test_split(df_known.index, test_size=test_ratio,
                                       stratify=strat, random_state=seed)
    strat_tv = df_known.loc[idx_tv,'y'] if strat is not None else None
    idx_tr, idx_va = train_test_split(idx_tv,
                                       test_size=val_ratio/(1-test_ratio),
                                       stratify=strat_tv, random_state=seed)

    print(f'  Train {len(idx_tr):,} | Val {len(idx_va):,} | Test {len(idx_te):,}')
    print(f'  ZD pool: {len(df_zd_full):,}')

    scaler  = RobustScaler()
    X_tr    = scaler.fit_transform(df_known.loc[idx_tr,feat_cols].values.astype(np.float32))
    X_va    = scaler.transform(df_known.loc[idx_va,feat_cols].values.astype(np.float32))
    X_te    = scaler.transform(df_known.loc[idx_te,feat_cols].values.astype(np.float32))
    X_zd    = scaler.transform(df_zd_full[feat_cols].values.astype(np.float32))

    clip = 10.
    X_tr, X_va, X_te, X_zd = [np.clip(x, -clip, clip) for x in [X_tr, X_va, X_te, X_zd]]
    X_tr, X_va, X_te, X_zd = [
        np.nan_to_num(x, nan=0.0, posinf=clip, neginf=-clip)
        for x in [X_tr, X_va, X_te, X_zd]
    ]

    y_tr = df_known.loc[idx_tr,'y'].values
    y_va = df_known.loc[idx_va,'y'].values
    y_te = df_known.loc[idx_te,'y'].values
    y_zd = df_zd_full['attack_cat'].values

    return dict(
        X_train=X_tr, y_train=y_tr,
        X_val=X_va,   y_val=y_va,
        X_test=X_te,  y_test=y_te,
        X_zd=X_zd,    y_zd=y_zd,
        n_features=len(feat_cols), n_classes=n_classes,
        label_encoder=le, scaler=scaler, feat_cols=feat_cols,
        categorical_maps=categorical_maps,
        known_cats=act_known, zd_cats=act_zd,
    )


# ═══════════════════════════════════════════════════════════════

class FlowDS(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.X[i], self.y[i]


def make_loaders(splits, batch_size=512, num_workers=2,
                 dos_class_idx=None, dos_over=5.0, seed=42):
    y_tr    = splits['y_train']
    freq    = np.bincount(y_tr)
    weights = 1.0 / freq[y_tr].astype(np.float32)
    if dos_class_idx is not None:
        weights[y_tr == dos_class_idx] *= dos_over

    tr_ds = FlowDS(splits['X_train'], y_tr)
    va_ds = FlowDS(splits['X_val'],   splits['y_val'])
    te_ds = FlowDS(splits['X_test'],  splits['y_test'])

    sampler = WeightedRandomSampler(torch.FloatTensor(weights), len(y_tr), replacement=True)

    gen = torch.Generator()
    gen.manual_seed(seed)

    def _seed_worker(worker_id):
        worker_seed = (seed + worker_id) % (2**32)
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    kw = dict(
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=_seed_worker,
        generator=gen,
    )
    return {
        'train': DataLoader(tr_ds, batch_size=batch_size, sampler=sampler, **kw),
        'val':   DataLoader(va_ds, batch_size=batch_size, shuffle=False, **kw),
        'test':  DataLoader(te_ds, batch_size=batch_size, shuffle=False, **kw),
    }


# ═══════════════════════════════════════════════════════════════
