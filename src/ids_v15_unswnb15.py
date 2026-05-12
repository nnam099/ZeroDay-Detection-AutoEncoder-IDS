"""
ids_v15_unswnb15.py — IDS v15.0  Hybrid Anomaly + GradBP + VAE + Attention
===========================================================================
Dataset : UNSW-NB15 (Kaggle)

CẢI TIẾN SO VỚI v14:
  [FIX-1]   Calibration đúng: threshold trên val-KNOWN chứ không trên toàn bộ val
  [FIX-2]   eval_epoch dùng normal_idx=0 thay vì hardcode; AUC macro multi-class
  [FIX-3]   ae_hidden được lưu đúng cách (không dùng enc[4].in_features)
  [FIX-4]   feature_names được lưu vào splits để không mất khi load lại
  [NEW-1]   VAE thay thế AE thuần → latent space có cấu trúc, ELBO loss
  [NEW-2]   Attention Gate trước classifier → trọng số feature động
  [NEW-3]   OOD Ensemble: ae_re + knn_distance + energy → vote majority
  [NEW-4]   Adaptive threshold: tự điều chỉnh thr theo từng class
  [NEW-5]   Save history dưới dạng JSON đầy đủ (không chỉ pth)
  [NEW-6]   Metric: thêm AUPRC (Average Precision) bên cạnh AUROC
  [NEW-7]   Config hỗ trợ YAML load (không override bằng CFG class)
  [TUNE-1]  Cosine Annealing with Warm Restarts thay vì LambdaLR đơn giản
  [TUNE-2]  Backbone: thêm ResBlock thứ 4 + skip connection cấp 2
  [TUNE-3]  Label smoothing động: giảm dần theo epoch
"""

import os, sys, glob, json, copy, time, pickle, warnings, random
if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    try: sys.stdout.reconfigure(encoding='utf-8')
    except Exception: pass
if sys.stderr and hasattr(sys.stderr, 'reconfigure'):
    try: sys.stderr.reconfigure(encoding='utf-8')
    except Exception: pass

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field, asdict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    classification_report, roc_curve, f1_score,
    confusion_matrix,
)
from sklearn.cluster import MiniBatchKMeans
from sklearn.neighbors import NearestNeighbors

warnings.filterwarnings('ignore')
torch.backends.cudnn.benchmark = True

# ═══════════════════════════════════════════════════════════════
# CONFIG — dùng dataclass thay vì class với class variables
# ═══════════════════════════════════════════════════════════════
@dataclass
class CFG:
    data_dir    : str   = '/kaggle/input'
    save_dir    : str   = '/kaggle/working/checkpoints_v15'
    plot_dir    : str   = '/kaggle/working/plots_v15'
    demo        : bool  = False

    # Training
    epochs      : int   = 100
    batch_size  : int   = 512
    lr          : float = 3e-4
    hidden      : int   = 256
    ae_hidden   : int   = 128
    latent_dim  : int   = 32       # [NEW] VAE latent dimension
    patience    : int   = 20

    # Loss
    lambda_con  : float = 0.3
    focal_gamma : float = 2.0
    dos_weight  : float = 8.0
    kl_weight   : float = 0.001    # [NEW] KL divergence weight cho VAE

    # Zero-day
    target_fpr        : float = 0.05
    n_clusters        : int   = 25
    n_neighbors_knn   : int   = 20  # [NEW] KNN OOD detector
    zd_augment_factor : int   = 1

    num_workers : int   = 2
    seed        : int   = 42


def resolve_paths(cfg: CFG) -> CFG:
    base_dir   = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    local_data = os.path.join(base_dir, 'data')
    local_ckpt = os.path.join(base_dir, 'checkpoints')
    local_plot = os.path.join(base_dir, 'plots')

    if getattr(cfg, 'data_dir', None):
        if (not os.path.exists(cfg.data_dir)) and os.path.exists(local_data):
            cfg.data_dir = local_data
        if str(cfg.data_dir).startswith('/kaggle/') and os.path.exists(local_data):
            cfg.data_dir = local_data

    if getattr(cfg, 'save_dir', None) and str(cfg.save_dir).startswith('/kaggle/'):
        cfg.save_dir = local_ckpt
    if getattr(cfg, 'plot_dir', None) and str(cfg.plot_dir).startswith('/kaggle/'):
        cfg.plot_dir = local_plot
    return cfg


def seed_everything(seed: int):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass


def _apply_yaml_config(cfg: CFG, data: Dict[str, Any]) -> CFG:
    if not isinstance(data, dict):
        return cfg
    section_map = {
        'training': ['epochs', 'batch_size', 'lr', 'patience', 'seed', 'num_workers'],
        'model': ['hidden', 'ae_hidden', 'latent_dim'],
        'loss': ['lambda_con', 'focal_gamma', 'dos_weight', 'kl_weight'],
        'detection': ['target_fpr', 'n_clusters', 'n_neighbors_knn', 'zd_augment_factor'],
        'paths': ['data_dir', 'save_dir', 'plot_dir'],
    }
    for section, keys in section_map.items():
        if section in data and isinstance(data[section], dict):
            for k in keys:
                if k in data[section]:
                    setattr(cfg, k, data[section][k])
    # Also allow flat keys
    for k, v in data.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    return cfg


def get_config() -> CFG:
    cfg = CFG()
    in_nb = False
    try:
        shell = get_ipython().__class__.__name__   # noqa
        in_nb = shell in ('ZMQInteractiveShell', 'Shell')
    except NameError:
        pass
    if in_nb:
        return resolve_paths(cfg)
    import argparse

    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument('--config', type=str, default=None)
    pre_args, _ = pre.parse_known_args()

    config_path = pre_args.config
    if not config_path:
        default_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'configs', 'config_default.yaml'))
        if os.path.exists(default_path):
            config_path = default_path

    if config_path:
        try:
            import yaml
            with open(config_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
            cfg = _apply_yaml_config(cfg, data)
        except ImportError:
            print('[WARN] pyyaml not installed — skip YAML config')
        except Exception as e:
            print(f'[WARN] YAML config load failed: {e}')

    p = argparse.ArgumentParser(description='IDS v15.0', parents=[pre])
    for fname, ftype in CFG.__dataclass_fields__.items():
        default = getattr(cfg, fname)
        if isinstance(default, bool):
            p.add_argument(f'--{fname}', action='store_true', default=default)
        else:
            p.add_argument(f'--{fname}', type=type(default), default=default)
    args = p.parse_args()
    for k, v in vars(args).items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    return resolve_paths(cfg)


# ═══════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════
KNOWN_ATTACK_CATS    = ['Normal','DoS','Exploits','Reconnaissance','Generic']
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
# DATA PIPELINE  (không thay đổi nhiều — đã ổn ở v14)
# ═══════════════════════════════════════════════════════════════
def _find_unsw_csvs(data_dir: str) -> List[str]:
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


def load_unsw_csvs(data_dir: str) -> pd.DataFrame:
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
                df = pd.read_csv(path, low_memory=False, encoding='utf-8', on_bad_lines='skip')
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


def normalize_labels(df: pd.DataFrame) -> pd.DataFrame:
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


def _encode_categorical_features(df: pd.DataFrame,
                                 categorical_maps: Optional[Dict[str, Dict[str, int]]] = None
                                 ) -> Dict[str, Dict[str, int]]:
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


def _get_numeric_features(df: pd.DataFrame,
                          categorical_maps: Optional[Dict[str, Dict[str, int]]] = None
                          ) -> Tuple[List[str], Dict[str, Dict[str, int]]]:
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


def engineer_features(df: pd.DataFrame, feat_cols: List[str]) -> Tuple[pd.DataFrame, List[str]]:
    """Feature engineering — thêm các đặc trưng mạng quan trọng."""
    eps = 1e-8
    new = []

    def add(name: str, vals):
        arr = np.asarray(vals, dtype=np.float32)
        arr = np.where(np.isfinite(arr), arr, 0.)
        df[name] = arr
        new.append(name)

    if 'sbytes' in df.columns and 'dbytes' in df.columns:
        tb = df['sbytes'] + df['dbytes'] + eps
        add('bytes_ratio',     df['sbytes'] / tb)
        add('log_total_bytes', np.log1p(tb))
        add('log_sbytes',      np.log1p(df['sbytes'].clip(lower=0)))
        add('log_dbytes',      np.log1p(df['dbytes'].clip(lower=0)))

    if 'spkts' in df.columns and 'dpkts' in df.columns:
        tp = df['spkts'] + df['dpkts'] + eps
        add('pkts_ratio',     df['spkts'] / tp)
        add('log_total_pkts', np.log1p(tp))
        # [NEW] bytes per packet
        if 'sbytes' in df.columns:
            add('bytes_per_pkt_src', df['sbytes'] / (df['spkts'] + eps))
        if 'dbytes' in df.columns:
            add('bytes_per_pkt_dst', df['dbytes'] / (df['dpkts'] + eps))

    if 'sbytes' in df.columns and 'dur' in df.columns:
        dur_s = df['dur'].clip(lower=1e-6)
        add('src_bps',     df['sbytes'] / dur_s)
        add('log_src_bps', np.log1p(df['sbytes'] / dur_s))
        add('pkt_rate',    (df.get('spkts', pd.Series(0, index=df.index)) + eps) / dur_s)

    if 'sload' in df.columns and 'dload' in df.columns:
        tl = df['sload'] + df['dload'] + eps
        add('load_asym', (df['sload'] - df['dload']).abs() / tl)
        add('log_sload', np.log1p(df['sload'].clip(lower=0)))
        add('log_dload', np.log1p(df['dload'].clip(lower=0)))  # [NEW]

    if 'sttl' in df.columns and 'dttl' in df.columns:
        add('ttl_diff', (df['sttl'] - df['dttl']).abs())
        add('ttl_sum',  df['sttl'] + df['dttl'])

    if 'sloss' in df.columns and 'spkts' in df.columns:
        add('loss_rate_src', df['sloss'] / (df['spkts'] + eps))
    if 'dloss' in df.columns and 'dpkts' in df.columns:
        add('loss_rate_dst', df['dloss'] / (df['dpkts'] + eps))

    if 'sjit' in df.columns and 'djit' in df.columns:
        add('jit_ratio', df['sjit'] / (df['djit'] + eps))
        add('log_sjit',  np.log1p(df['sjit'].clip(lower=0)))
        add('jit_sum',   df['sjit'] + df['djit'])  # [NEW]

    if 'synack' in df.columns and 'ackdat' in df.columns:
        add('handshake_ratio', df['synack'] / (df['ackdat'] + eps))
        add('incomplete_tcp',  ((df['synack'] > 0) & (df['ackdat'] == 0)).astype(float))

    if 'sintpkt' in df.columns and 'dintpkt' in df.columns:
        add('intpkt_ratio', df['sintpkt'] / (df['dintpkt'] + eps))
        add('intpkt_diff',  (df['sintpkt'] - df['dintpkt']).abs())  # [NEW]

    # [NEW] Connection count features ratio
    if 'ct_srv_src' in df.columns and 'ct_srv_dst' in df.columns:
        add('ct_srv_ratio', df['ct_srv_src'] / (df['ct_srv_dst'] + eps))

    return df, feat_cols + new


def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.replace([np.inf, -np.inf], np.nan)
    n = len(df)
    df = df.drop_duplicates()
    print(f'  Removed {n - len(df):,} duplicates')
    return df


def prepare_splits(df: pd.DataFrame,
                   known_cats=KNOWN_ATTACK_CATS,
                   zd_cats=ZERO_DAY_ATTACK_CATS,
                   test_ratio: float = 0.20,
                   val_ratio: float  = 0.10,
                   seed: int         = 42,
                   zd_augment: int   = 1) -> Dict[str, Any]:
    print('\n[DATA SPLIT]')
    df = normalize_labels(df)

    cat_counts = df['attack_cat'].value_counts()
    print('\n  Distribution:')
    for cat, cnt in cat_counts.items():
        m = 'K' if cat in known_cats else ('Z' if cat in zd_cats else '?')
        print(f'    [{m}] {cat:<22} {cnt:>8,}')

    avail     = set(df['attack_cat'].unique())
    act_known = [c for c in known_cats if c in avail]
    act_zd    = [c for c in zd_cats    if c in avail]

    for cat in avail - set(act_known) - set(act_zd):
        n = (df['attack_cat'] == cat).sum()
        if n < 5000: act_zd.append(cat)
        else:        act_known.append(cat)

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
    strat_tv = df_known.loc[idx_tv, 'y'] if strat is not None else None
    idx_tr, idx_va = train_test_split(idx_tv,
                                       test_size=val_ratio / (1 - test_ratio),
                                       stratify=strat_tv, random_state=seed)

    print(f'  Train {len(idx_tr):,} | Val {len(idx_va):,} | Test {len(idx_te):,}')
    print(f'  ZD pool: {len(df_zd_full):,}')

    scaler = RobustScaler()
    X_tr   = scaler.fit_transform(df_known.loc[idx_tr, feat_cols].values.astype(np.float32))
    X_va   = scaler.transform(df_known.loc[idx_va, feat_cols].values.astype(np.float32))
    X_te   = scaler.transform(df_known.loc[idx_te, feat_cols].values.astype(np.float32))
    X_zd   = scaler.transform(df_zd_full[feat_cols].values.astype(np.float32))

    clip = 10.
    X_tr, X_va, X_te, X_zd = [np.clip(x, -clip, clip) for x in [X_tr, X_va, X_te, X_zd]]
    X_tr, X_va, X_te, X_zd = [
        np.nan_to_num(x, nan=0.0, posinf=clip, neginf=-clip)
        for x in [X_tr, X_va, X_te, X_zd]
    ]

    y_tr = df_known.loc[idx_tr, 'y'].values
    y_va = df_known.loc[idx_va, 'y'].values
    y_te = df_known.loc[idx_te, 'y'].values
    y_zd = df_zd_full['attack_cat'].values

    return dict(
        X_train=X_tr, y_train=y_tr,
        X_val=X_va,   y_val=y_va,
        X_test=X_te,  y_test=y_te,
        X_zd=X_zd,    y_zd=y_zd,
        n_features=len(feat_cols), n_classes=n_classes,
        label_encoder=le, scaler=scaler,
        feat_cols=feat_cols, feature_names=feat_cols,  # [FIX-4] lưu cả hai
        categorical_maps=categorical_maps,
        known_cats=act_known, zd_cats=act_zd,
    )


# ═══════════════════════════════════════════════════════════════
# MODEL v15 — VAE + Attention Gate + Deeper Backbone
# ═══════════════════════════════════════════════════════════════

class ResBlock(nn.Module):
    """ResBlock với Pre-LN (ổn định hơn khi train sâu)."""
    def __init__(self, dim: int, dropout: float = 0.2):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-LN residual connection
        return x + self.ff(self.norm2(self.norm1(x)))


class AttentionGate(nn.Module):
    """[NEW] Feature attention — model tự học feature nào quan trọng."""
    def __init__(self, dim: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.Tanh(),
            nn.Linear(dim // 4, dim),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        w = self.gate(x)
        return x * w, w  # trả về cả attention weight để visualize


class IDSBackbone(nn.Module):
    """Backbone với 4 ResBlocks + 2-level skip + Attention Gate."""
    def __init__(self, n_features: int, hidden: int = 256):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
        )
        self.res1 = ResBlock(hidden)
        self.res2 = ResBlock(hidden)
        self.res3 = ResBlock(hidden)
        self.res4 = ResBlock(hidden)                  # [NEW] thêm block 4
        self.attn_gate = AttentionGate(hidden)        # [NEW]
        self.hidden = hidden

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h0 = self.input_proj(x)
        h1 = self.res1(h0)
        h2 = self.res2(h1)
        h3 = self.res3(h2 + h0)   # [NEW] 2-level skip
        h4 = self.res4(h3)
        h_att, attn_w = self.attn_gate(h4)
        return h_att, attn_w


class VAE(nn.Module):
    """[NEW] Variational AutoEncoder — ELBO loss → latent space có cấu trúc hơn AE.

    Ưu điểm so với AE thuần:
    - Latent space smooth → interpolation tốt hơn
    - KL divergence buộc z ~ N(0,1) → OOD detection dựa trên z đáng tin cậy hơn
    - Reconstruction error vẫn là anomaly score chính
    """
    def __init__(self, n_features: int, ae_hidden: int = 128, latent_dim: int = 32):
        super().__init__()
        mid = ae_hidden
        self.latent_dim = latent_dim

        # Encoder → μ, log σ²
        self.enc = nn.Sequential(
            nn.Linear(n_features, mid * 2), nn.LayerNorm(mid * 2), nn.GELU(),
            nn.Linear(mid * 2, mid),        nn.LayerNorm(mid),     nn.GELU(),
        )
        self.fc_mu     = nn.Linear(mid, latent_dim)
        self.fc_logvar = nn.Linear(mid, latent_dim)

        # Decoder
        self.dec = nn.Sequential(
            nn.Linear(latent_dim, mid),     nn.LayerNorm(mid),     nn.GELU(),
            nn.Linear(mid, mid * 2),        nn.LayerNorm(mid * 2), nn.GELU(),
            nn.Linear(mid * 2, n_features),
        )

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h      = self.enc(x)
        mu     = self.fc_mu(h)
        logvar = self.fc_logvar(h).clamp(-10, 2)   # clamp để tránh NaN
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if self.training:
            std = (0.5 * logvar).exp()
            return mu + std * torch.randn_like(std)
        return mu   # eval time: dùng mean

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.dec(z)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z          = self.reparameterize(mu, logvar)
        x_hat      = self.decode(z)
        return x_hat, mu, logvar

    def recon_error(self, x: torch.Tensor) -> torch.Tensor:
        """MSE reconstruction error per sample — anomaly score."""
        x_hat, _, _ = self.forward(x)
        return (x_hat - x).pow(2).mean(dim=-1)

    def elbo_loss(self, x: torch.Tensor, kl_weight: float = 0.001) -> torch.Tensor:
        """ELBO = Reconstruction Loss + β·KL divergence."""
        x_hat, mu, logvar = self.forward(x)
        recon = F.mse_loss(x_hat, x, reduction='mean')
        # KL(q(z|x) || N(0,I))
        kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=-1).mean()
        return recon + kl_weight * kl


class IDSModel(nn.Module):
    """Model chính v15: Backbone + Attention + VAE + Proj Head."""
    def __init__(self, n_features: int, n_classes: int,
                 hidden: int = 256, ae_hidden: int = 128, latent_dim: int = 32):
        super().__init__()
        self.n_classes  = n_classes
        self.n_features = n_features
        self.hidden_dim = hidden      # [FIX-3] lưu trực tiếp
        self.ae_hidden_dim = ae_hidden

        self.backbone   = IDSBackbone(n_features, hidden)
        self.classifier = nn.Linear(hidden, n_classes)
        self.proj_head  = nn.Sequential(
            nn.Linear(hidden, 128), nn.GELU(),
            nn.Linear(128, 64),
        )
        self.vae        = VAE(n_features, ae_hidden, latent_dim)  # [NEW] VAE
        self.log_temp   = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        fv, _  = self.backbone(x)
        logits = self.classifier(fv)
        return logits, fv

    def get_embed(self, x: torch.Tensor) -> torch.Tensor:
        fv, _ = self.backbone(x)
        return F.normalize(self.proj_head(fv), dim=-1)

    def get_attention(self, x: torch.Tensor) -> torch.Tensor:
        """Trả về attention weights — dùng để visualize feature importance."""
        _, attn_w = self.backbone(x)
        return attn_w

    def energy_score(self, x: torch.Tensor) -> torch.Tensor:
        logits, _ = self.forward(x)
        T = self.log_temp.exp().clamp(0.5, 5.0)
        return -T * torch.logsumexp(logits / T, dim=-1)

    def gradbp_score(self, x: torch.Tensor) -> torch.Tensor:
        x  = x.detach()
        fv, _ = self.backbone(x)
        fv.requires_grad_(True)
        logits = self.classifier(fv)
        pred   = logits.argmax(dim=-1)
        loss   = F.cross_entropy(logits, pred)
        grad   = torch.autograd.grad(loss, fv)[0]
        return grad.norm(dim=-1)

    def fv_cluster_score(self, x: torch.Tensor, centroids: torch.Tensor) -> torch.Tensor:
        _, fv  = self.forward(x)
        fv_n   = F.normalize(fv, dim=-1)
        cent_n = F.normalize(centroids, dim=-1)
        return torch.cdist(fv_n, cent_n).min(dim=-1).values

    def hybrid_score(self, x: torch.Tensor, w_ae: float = 0.5, w_cls: float = 0.5) -> torch.Tensor:
        re    = self.vae.recon_error(x)
        probs = torch.softmax(self.forward(x)[0], dim=-1)
        cls_s = 1 - probs.max(dim=-1).values
        return w_ae * re + w_cls * cls_s


# ═══════════════════════════════════════════════════════════════
# LOSS v15 — FocalLoss + SupCon + VAE ELBO
# ═══════════════════════════════════════════════════════════════
class FocalLoss(nn.Module):
    def __init__(self, n_classes: int, gamma: float = 2.5,
                 label_smooth: float = 0.05, class_weights=None):
        super().__init__()
        self.gamma  = gamma
        self.ls     = label_smooth
        self.nc     = n_classes
        self.w      = class_weights

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        ce  = F.cross_entropy(logits, labels, reduction='none',
                              label_smoothing=self.ls, weight=self.w)
        pt  = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()


class SupConLoss(nn.Module):
    def __init__(self, T: float = 0.15):
        super().__init__()
        self.T = T

    def forward(self, feats: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        B    = feats.shape[0]
        sim  = torch.matmul(feats, feats.T) / self.T
        sim  = sim - sim.detach().max()
        lmat = labels.view(-1, 1)
        pos  = (lmat == lmat.T).float().fill_diagonal_(0)
        eye  = torch.eye(B, device=feats.device, dtype=torch.bool)
        denom= (torch.exp(sim) * (~eye).float()).sum(1, keepdim=True).clamp(min=1e-8)
        lp   = sim - torch.log(denom)
        cnt  = pos.sum(1).clamp(min=1e-8)
        loss = -(pos * lp).sum(1) / cnt
        valid = pos.sum(1) > 0
        if not valid.any():
            return torch.tensor(0.0, device=feats.device, requires_grad=True)
        return loss[valid].mean()


class IDSLoss(nn.Module):
    def __init__(self, n_classes: int, lambda_con: float = 0.3,
                 focal_gamma: float = 2.0, dos_class_idx: Optional[int] = None,
                 dos_weight: float = 5.0, kl_weight: float = 0.001, device: str = 'cpu'):
        super().__init__()
        w = torch.ones(n_classes, device=device)
        if dos_class_idx is not None:
            w[dos_class_idx] = dos_weight
        w = w / w.mean()
        self.focal = FocalLoss(n_classes, gamma=focal_gamma, label_smooth=0.05,
                               class_weights=w.to(device))
        self.con      = SupConLoss(T=0.15)
        self.lam      = lambda_con
        self.kl_w     = kl_weight   # [NEW] VAE KL weight

    def forward(self, logits: torch.Tensor, labels: torch.Tensor,
                embeds: torch.Tensor, x: torch.Tensor,
                model_vae) -> Dict[str, torch.Tensor]:
        lf   = self.focal(logits, labels)
        lc   = self.con(embeds, labels)
        lae  = model_vae.elbo_loss(x, self.kl_w)   # [NEW] ELBO thay MSE đơn thuần
        lf   = torch.nan_to_num(lf,  nan=0.0, posinf=10.0)
        lc   = torch.nan_to_num(lc,  nan=0.0, posinf=10.0)
        lae  = torch.nan_to_num(lae, nan=0.0, posinf=10.0)
        total = lf + self.lam * lc + 0.5 * lae
        return {'total': total, 'focal': lf, 'con': lc, 'vae': lae}


# ═══════════════════════════════════════════════════════════════
# DATASET / LOADER
# ═══════════════════════════════════════════════════════════════
class FlowDS(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)

    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.X[i], self.y[i]


def make_loaders(splits: Dict, batch_size: int = 512, num_workers: int = 2,
                 dos_class_idx: Optional[int] = None, dos_over: float = 5.0,
                 seed: int = 42) -> Dict:
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
# TRAINING
# ═══════════════════════════════════════════════════════════════
def train_epoch(model: IDSModel, loader: DataLoader, optimizer: torch.optim.Optimizer,
                criterion: IDSLoss, device: str) -> Dict[str, float]:
    model.train()
    tot_loss = ncorr = ntot = 0
    use_amp  = (device == 'cuda')
    amp_scaler = torch.amp.GradScaler('cuda') if use_amp else None

    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)

        if use_amp:
            with torch.amp.autocast('cuda'):
                logits, _ = model(X)
                embeds    = model.get_embed(X)
                losses    = criterion(logits, y, embeds, X, model.vae)
                loss      = losses['total']
            amp_scaler.scale(loss).backward()
            amp_scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            amp_scaler.step(optimizer)
            amp_scaler.update()
        else:
            logits, _ = model(X)
            embeds    = model.get_embed(X)
            losses    = criterion(logits, y, embeds, X, model.vae)
            loss      = losses['total']
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        loss_val = loss.item()
        if not np.isfinite(loss_val):
            continue
        tot_loss += loss_val * len(y)
        ncorr    += (logits.argmax(1) == y).sum().item()
        ntot     += len(y)

    if ntot == 0:
        return {'loss': float('nan'), 'acc': 0.0}
    return {'loss': tot_loss / ntot, 'acc': ncorr / ntot}


@torch.no_grad()
def eval_epoch(model: IDSModel, loader: DataLoader, device: str,
               normal_idx: int = 0) -> Dict[str, float]:
    """[FIX-2] Dùng normal_idx từ label_encoder, không hardcode."""
    model.eval()
    all_probs, all_labels = [], []
    for X, y in loader:
        logits, _ = model(X.to(device))
        all_probs.append(torch.softmax(logits, dim=-1).cpu().numpy())
        all_labels.append(y.numpy())
    probs  = np.concatenate(all_probs)
    labels = np.concatenate(all_labels)
    binary = (labels != normal_idx).astype(int)
    score  = 1 - probs[:, normal_idx]
    try:    auc = roc_auc_score(binary, score)
    except: auc = 0.5
    return {'auc': auc, 'acc': (probs.argmax(1) == labels).mean()}


def train(model: IDSModel, loaders: Dict, args: CFG,
          criterion: IDSLoss, device: str,
          normal_idx: int = 0) -> Tuple[IDSModel, List[Dict]]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # [TUNE-1] CosineAnnealingWarmRestarts thay LambdaLR
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=20, T_mult=2, eta_min=1e-6,
    )

    best_auc = 0.
    best_sd  = None
    patience = 0
    history  = []

    print(f'\n{"="*65}')
    print(f'Training v15 | epochs={args.epochs} lr={args.lr} hidden={args.hidden}')
    print(f'{"="*65}')

    for ep in range(1, args.epochs + 1):
        t0  = time.time()
        trm = train_epoch(model, loaders['train'], optimizer, criterion, device)
        vam = eval_epoch(model,  loaders['val'],   device, normal_idx)
        scheduler.step(ep)

        is_best = vam['auc'] > best_auc
        if is_best:
            best_auc = vam['auc']
            best_sd  = copy.deepcopy(model.state_dict())
            patience = 0
        else:
            patience += 1

        lr_now = optimizer.param_groups[0]['lr']
        history.append({**trm, **{f'val_{k}': v for k, v in vam.items()}, 'lr': lr_now})
        star = '*' if is_best else ' '
        print(f'{star}[{ep:3d}/{args.epochs}] loss={trm["loss"]:.4f} acc={trm["acc"]:.4f}'
              f'  vAUC={vam["auc"]:.4f}  lr={lr_now:.2e}  P={patience}  ({time.time()-t0:.1f}s)')

        if patience >= args.patience:
            print(f'  Early stop at ep {ep}')
            break

    if best_sd:
        model.load_state_dict(best_sd)
    print(f'\nBest Val AUC = {best_auc:.4f}')
    return model, history


# ═══════════════════════════════════════════════════════════════
# SCORING HELPERS v15
# ═══════════════════════════════════════════════════════════════
def _batch_scores(model: IDSModel, X: np.ndarray, device: str,
                  centroids=None, batch: int = 512) -> Dict[str, np.ndarray]:
    model.eval()
    s_e, s_sm, s_fvc, s_re, s_hyb = [], [], [], [], []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            x  = torch.FloatTensor(X[i:i+batch]).to(device)
            s_e.append(model.energy_score(x).cpu().numpy())
            probs = torch.softmax(model.forward(x)[0], dim=-1)
            s_sm.append((1 - probs.max(dim=-1).values).cpu().numpy())
            re = model.vae.recon_error(x)    # [NEW] VAE recon error
            s_re.append(re.cpu().numpy())
            if centroids is not None:
                s_fvc.append(model.fv_cluster_score(x, centroids).cpu().numpy())
                hyb = 0.5 * re + 0.5 * (1 - probs.max(dim=-1).values)
                s_hyb.append(hyb.cpu().numpy())
    out = {
        'energy':  np.concatenate(s_e),
        'softmax': np.concatenate(s_sm),
        'ae_re':   np.concatenate(s_re),
    }
    if centroids is not None:
        out['fv_cluster'] = np.concatenate(s_fvc)
        out['hybrid']     = np.concatenate(s_hyb)
    return out


def _batch_gradbp(model: IDSModel, X: np.ndarray, device: str, batch: int = 512) -> np.ndarray:
    scores = []
    for i in range(0, len(X), batch):
        x = torch.FloatTensor(X[i:i+batch]).to(device)
        scores.append(model.gradbp_score(x).detach().cpu().numpy())
    return np.concatenate(scores)


def build_knn_detector(model: IDSModel, X_train: np.ndarray,
                        n_neighbors: int = 20, device: str = 'cpu') -> NearestNeighbors:
    """[NEW] KNN trong feature space — distance to k-NN là anomaly score."""
    model.eval()
    fvs = []
    with torch.no_grad():
        for i in range(0, len(X_train), 1024):
            x = torch.FloatTensor(X_train[i:i+1024]).to(device)
            _, fv = model(x)
            fvs.append(F.normalize(fv, dim=-1).cpu().numpy())
    all_fvs = np.nan_to_num(np.concatenate(fvs), nan=0.0)
    knn = NearestNeighbors(n_neighbors=n_neighbors, metric='cosine', algorithm='auto', n_jobs=-1)
    knn.fit(all_fvs)
    print(f'  KNN detector fitted: {len(all_fvs):,} samples, k={n_neighbors}')
    return knn


def _batch_knn_score(model: IDSModel, X: np.ndarray, knn: NearestNeighbors,
                     device: str, batch: int = 512) -> np.ndarray:
    """Trả về mean distance đến k nearest neighbors."""
    model.eval()
    fvs = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            x = torch.FloatTensor(X[i:i+batch]).to(device)
            _, fv = model(x)
            fvs.append(F.normalize(fv, dim=-1).cpu().numpy())
    all_fvs = np.nan_to_num(np.concatenate(fvs), nan=0.0)
    dists, _ = knn.kneighbors(all_fvs)
    return dists.mean(axis=1)


def build_centroids(model: IDSModel, X_tr: np.ndarray, y_tr: np.ndarray,
                    n_clusters: int = 25, device: str = 'cpu',
                    seed: int = 42) -> torch.Tensor:
    model.eval()
    fvs = []
    with torch.no_grad():
        for i in range(0, len(X_tr), 1024):
            x = torch.FloatTensor(X_tr[i:i+1024]).to(device)
            _, fv = model(x)
            fv_np = F.normalize(fv, dim=-1).cpu().float().numpy()
            fvs.append(fv_np)
    all_fvs = np.nan_to_num(np.concatenate(fvs), nan=0.0)
    centers = []
    for cls in range(len(np.unique(y_tr))):
        m  = y_tr == cls
        cv = all_fvs[m]
        if len(cv) == 0: continue
        cv = cv[np.isfinite(cv).all(axis=1)]
        if len(cv) == 0: continue
        k = min(n_clusters, len(cv))
        if k == 1:
            centers.append(cv.mean(0, keepdims=True))
        else:
            km = MiniBatchKMeans(n_clusters=k, random_state=seed, n_init=3, batch_size=2048)
            centers.append(km.fit(cv).cluster_centers_)
    c = np.concatenate(centers)
    print(f'  Centroids: {len(c)}')
    return torch.FloatTensor(c).to(device)


def calibrate(model: IDSModel, X_val: np.ndarray, y_val: np.ndarray,
              target_fpr: float, device: str, centroids,
              knn: Optional[NearestNeighbors] = None) -> Dict[str, float]:
    """[FIX-1] Chỉ dùng known-class samples để set threshold."""
    scores = _batch_scores(model, X_val, device, centroids)
    scores['gradbp_l2'] = _batch_gradbp(model, X_val, device)
    if knn is not None:
        scores['knn_dist'] = _batch_knn_score(model, X_val, knn, device)

    thr = {}
    print(f'\n  Thresholds @ FPR={target_fpr * 100:.0f}%  (calibrated on KNOWN val)')
    for m, arr in scores.items():
        t = float(np.quantile(arr, 1.0 - target_fpr))
        thr[m] = t
        print(f'    {m:<16} thr={t:.6f}  actual_FPR={(arr > t).mean():.4f}')
    return thr


# ═══════════════════════════════════════════════════════════════
# EVALUATION v15
# ═══════════════════════════════════════════════════════════════
def evaluate_classifier(model: IDSModel, X_te: np.ndarray, y_te: np.ndarray,
                         label_names: List[str], device: str) -> Dict:
    model.eval()
    preds, probs_list = [], []
    with torch.no_grad():
        for i in range(0, len(X_te), 512):
            x = torch.FloatTensor(X_te[i:i+512]).to(device)
            lg, _ = model(x)
            preds.append(lg.argmax(1).cpu().numpy())
            probs_list.append(torch.softmax(lg, dim=-1).cpu().numpy())
    preds = np.concatenate(preds)
    probs = np.concatenate(probs_list)
    print(classification_report(y_te, preds, target_names=label_names, digits=4))

    ni     = label_names.index('Normal') if 'Normal' in label_names else 0
    bin_   = (y_te != ni).astype(int)
    score  = 1 - probs[:, ni]
    try:    auc  = roc_auc_score(bin_, score)
    except: auc  = 0.5
    try:    aupr = average_precision_score(bin_, score)  # [NEW] AUPRC
    except: aupr = 0.5
    print(f'  AUC(Normal vs Attack) : {auc:.4f}')
    print(f'  AUPRC(Normal vs Attack): {aupr:.4f}')
    return {'preds': preds, 'probs': probs, 'auc': auc, 'auprc': aupr}


def evaluate_zero_day(model: IDSModel, X_kn: np.ndarray, y_kn: np.ndarray,
                       X_zd: np.ndarray, y_zd: np.ndarray,
                       thr: Dict, centroids, device: str,
                       knn: Optional[NearestNeighbors] = None) -> Dict:
    print(f'\n{"="*65}')
    print(f'ZERO-DAY DETECTION  |  Known={len(X_kn):,}  ZD={len(X_zd):,}')
    print(f'{"="*65}')
    sk = _batch_scores(model, X_kn, device, centroids)
    sz = _batch_scores(model, X_zd, device, centroids)
    sk['gradbp_l2'] = _batch_gradbp(model, X_kn, device)
    sz['gradbp_l2'] = _batch_gradbp(model, X_zd, device)

    if knn is not None:
        sk['knn_dist'] = _batch_knn_score(model, X_kn, knn, device)
        sz['knn_dist'] = _batch_knn_score(model, X_zd, knn, device)

    # [NEW] OOD Ensemble: vote majority giữa ae_re, knn_dist, hybrid
    ens_methods = ['ae_re', 'hybrid']
    if 'knn_dist' in sk:
        ens_methods.append('knn_dist')
    if len(ens_methods) >= 2:
        ens_k = np.zeros(len(X_kn))
        ens_z = np.zeros(len(X_zd))
        for m in ens_methods:
            t = thr.get(m, float(np.quantile(sk[m], 0.95)))
            ens_k += (sk[m] > t).astype(float)
            ens_z += (sz[m] > t).astype(float)
        sk['ood_ensemble'] = ens_k / len(ens_methods)
        sz['ood_ensemble'] = ens_z / len(ens_methods)

    true    = np.concatenate([np.zeros(len(X_kn)), np.ones(len(X_zd))])
    results = {}
    print(f'\n  {"Method":<16} {"AUC":>8} {"AUPRC":>8} {"TPR@1%":>10} {"TPR@5%":>10}')
    print(f'  {"-"*56}')
    method_order = ['ood_ensemble','gradbp_l2','hybrid','ae_re','energy','softmax',
                    'fv_cluster','knn_dist']
    for m in method_order:
        if m not in sk: continue
        sc_all = np.concatenate([sk[m], sz[m]])
        try:    auc  = roc_auc_score(true, sc_all)
        except: auc  = 0.5
        try:    aupr = average_precision_score(true, sc_all)  # [NEW]
        except: aupr = 0.5
        fpr_a, tpr_a, _ = roc_curve(true, sc_all)
        def tpr_at(tfpr):
            idx = np.searchsorted(fpr_a, tfpr)
            return float(tpr_a[min(idx, len(tpr_a) - 1)])
        t1, t5 = tpr_at(0.01), tpr_at(0.05)
        print(f'  {m:<16} {auc:>8.4f} {aupr:>8.4f} {t1:>10.4f} {t5:>10.4f}')
        results[m] = {'auc': auc, 'auprc': aupr, 'tpr_1': t1, 'tpr_5': t5,
                      'fpr': fpr_a.tolist(), 'tpr': tpr_a.tolist(),
                      's_known': sk[m], 's_zd': sz[m]}

    best = max(results, key=lambda m: results[m]['auc'])
    bt   = thr.get(best, float(np.quantile(sk[best], 0.95)))
    print(f'\n  Per-class recall [{best}@thr={bt:.5f}]:')
    per_cls = {}
    for cls in np.unique(y_zd):
        mask = y_zd == cls
        n    = mask.sum()
        det  = (sz[best][mask] > bt).sum()
        r    = det / n if n > 0 else 0.
        per_cls[str(cls)] = {'n': int(n), 'recall': r}
        bar  = '#' * int(r * 20) + '.' * (20 - int(r * 20))
        print(f'    {str(cls):<30} [{bar}] {r:.1%}  (n={n:,})')

    results['_per_class']    = per_cls
    results['_best_method']  = best
    results['_scores_known'] = sk
    results['_scores_zd']    = sz
    return results


# ═══════════════════════════════════════════════════════════════
# PLOTS
# ═══════════════════════════════════════════════════════════════
def plot_training_curve(history: List[Dict], save_path: str):
    epochs = list(range(1, len(history) + 1))
    fig, axes = plt.subplots(1, 4, figsize=(20, 4))  # [NEW] thêm LR panel
    fig.patch.set_facecolor('#0d1117')

    data_list = [
        [h['loss']    for h in history],
        [h['val_auc'] for h in history],
        [h['val_acc'] for h in history],
        [h.get('lr', 0) for h in history],
    ]
    titles  = ['Train Loss', 'Val AUC', 'Val Accuracy', 'Learning Rate']
    colors  = ['#FF6B6B', '#00BFFF', '#FFA500', '#90EE90']

    for ax, d, t, c in zip(axes, data_list, titles, colors):
        ax.set_facecolor('#0d1117')
        ax.plot(epochs, d, c=c, lw=2)
        ax.set_title(t, color='white', fontsize=12)
        ax.tick_params(colors='white')
        ax.set_xlabel('Epoch', color='white')
        for sp in ax.spines.values(): sp.set_edgecolor('#333')

    fig.suptitle('V15.0 — Training Curves', color='white', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='#0d1117')
    plt.close()
    print(f'  [Plot] Training curve → {save_path}')


def plot_soc_decision_space(model: IDSModel, X_kn: np.ndarray, y_kn_labels: np.ndarray,
                             X_zd: np.ndarray, p_thr: float, re_thr: float,
                             device: str, save_path: str,
                             label_names: List[str], n_sample: int = 5000,
                             seed: int = 42):
    model.eval()
    rng = np.random.default_rng(seed)

    def _sample(X, n):
        idx = rng.choice(len(X), min(n, len(X)), replace=False)
        return X[idx], idx

    X_kn_s, idx_kn = _sample(X_kn, n_sample)
    X_zd_s, _      = _sample(X_zd, n_sample // 2)

    with torch.no_grad():
        def prob_attack(X):
            probs = []
            for i in range(0, len(X), 512):
                x  = torch.FloatTensor(X[i:i+512]).to(device)
                lg, _ = model(x)
                p  = torch.softmax(lg, dim=-1)
                probs.append((1 - p[:, 0]).cpu().numpy())
            return np.concatenate(probs)

        def recon_err(X):
            res = []
            for i in range(0, len(X), 512):
                x = torch.FloatTensor(X[i:i+512]).to(device)
                res.append(model.vae.recon_error(x).cpu().numpy())
            return np.concatenate(res)

    p_kn = prob_attack(X_kn_s)
    r_kn = recon_err(X_kn_s)
    p_zd = prob_attack(X_zd_s)
    r_zd = recon_err(X_zd_s)

    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor('#0d1117')
    ax.set_facecolor('#0d1117')

    cmap = plt.cm.get_cmap('tab10', len(label_names))
    y_sub = y_kn_labels[idx_kn]
    for ci, name in enumerate(label_names):
        m = y_sub == ci
        if m.sum() == 0: continue
        ax.scatter(p_kn[m], r_kn[m], s=6, alpha=0.5, color=cmap(ci), label=name)
    ax.scatter(p_zd, r_zd, s=8, alpha=0.6, color='#FF4500', marker='x', label='Zero-Day')
    ax.axvline(p_thr,  color='#FFD700', linestyle='--', lw=1.2, label=f'P-Thr {p_thr:.2f}')
    ax.axhline(re_thr, color='white',   linestyle=':',  lw=1.2, label=f'RE-Thr {re_thr:.3f}')

    ax.set_yscale('log')
    ax.set_xlabel('P(Attack) — Classifier', color='white', fontsize=12)
    ax.set_ylabel('VAE Reconstruction Error', color='white', fontsize=12)
    ax.set_title('V15.0 — SOC Decision Space (VAE)', color='white', fontsize=14, fontweight='bold')
    ax.tick_params(colors='white')
    for sp in ax.spines.values(): sp.set_edgecolor('#444')
    ax.legend(facecolor='#1a1f2e', edgecolor='#555', labelcolor='white', fontsize=9, loc='lower right')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='#0d1117')
    plt.close()
    print(f'  [Plot] SOC Decision Space → {save_path}')


def plot_per_class_proper(label_names, y_test, preds, per_cls_zd, zd_classes_order,
                           save_path, normal_idx=0):
    known_names, known_recalls = [], []
    for i, name in enumerate(label_names):
        mask = y_test == i
        if mask.sum() == 0: continue
        det = (preds[mask] != normal_idx).mean() * 100
        known_names.append(name.upper())
        known_recalls.append(det)

    zd_names   = [c.upper() for c in zd_classes_order]
    zd_recalls = [per_cls_zd[c]['recall'] * 100 for c in zd_classes_order]

    fig, axes = plt.subplots(1, 2, figsize=(18, max(7, max(len(known_names), len(zd_names)) * 0.7)))
    fig.patch.set_facecolor('#0d1117')

    def draw_panel(ax, names, vals, color, title):
        ax.set_facecolor('#0d1117')
        bars = ax.barh(names, vals, color=color, height=0.6)
        for bar, val in zip(bars, vals):
            ax.text(min(bar.get_width() + 1, 108),
                    bar.get_y() + bar.get_height() / 2,
                    f'{val:.1f}%', va='center', color='white', fontsize=10)
        ax.set_xlim(0, 115)
        ax.axvline(95, color='#FFD700', linestyle='--', lw=1, alpha=0.7)
        ax.set_xlabel('Detection Rate (%)', color='white', fontsize=11)
        ax.set_title(title, color='white', fontsize=11)
        ax.tick_params(colors='white')
        for sp in ax.spines.values(): sp.set_edgecolor('#333')

    draw_panel(axes[0], known_names, known_recalls, '#00A8E8',
               'Known Attacks — Supervised Head (%)')
    draw_panel(axes[1], zd_names, zd_recalls, '#FFA500',
               'Zero-Day Attacks — OOD Ensemble (%)')

    fig.suptitle('V15.0 — Per-Class Detection Performance',
                 color='white', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='#0d1117')
    plt.close()
    print(f'  [Plot] Per-class → {save_path}')


def plot_roc_curves(zd_results: Dict, save_path: str):
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))  # [NEW] ROC + AUPRC
    fig.patch.set_facecolor('#0d1117')
    for ax in axes:
        ax.set_facecolor('#0d1117')

    method_colors = {
        'gradbp_l2':'#FF6B6B', 'hybrid':'#FFA500', 'ae_re':'#00BFFF',
        'energy':'#90EE90',    'softmax':'#DDA0DD', 'fv_cluster':'#F0E68C',
        'knn_dist':'#FF69B4',  'ood_ensemble':'#FFFFFF',
    }
    for m, r in zd_results.items():
        if m.startswith('_') or 'fpr' not in r: continue
        fpr = np.array(r['fpr']); tpr = np.array(r['tpr'])
        auc  = r['auc']
        aupr = r.get('auprc', 0.)
        c    = method_colors.get(m, 'gray')
        lw   = 2.5 if m == 'ood_ensemble' else 1.5
        ls   = '--' if m in ('energy', 'softmax') else '-'
        axes[0].plot(fpr, tpr, c=c, lw=lw, ls=ls, label=f'{m}  AUC={auc:.4f}')
        axes[1].plot(fpr, tpr, c=c, lw=lw, ls=ls, label=f'{m}  AUPRC={aupr:.4f}')

    for ax, title in zip(axes, ['ROC Curve (AUC)', 'Precision-Recall Proxy']):
        ax.plot([0, 1], [0, 1], '--', color='#555', lw=1)
        ax.set_xlabel('False Positive Rate', color='white', fontsize=11)
        ax.set_ylabel('True Positive Rate',  color='white', fontsize=11)
        ax.set_title(f'V15.0 — Zero-Day {title}', color='white', fontsize=12, fontweight='bold')
        ax.tick_params(colors='white')
        for sp in ax.spines.values(): sp.set_edgecolor('#333')
        ax.legend(facecolor='#1a1f2e', edgecolor='#555', labelcolor='white', fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='#0d1117')
    plt.close()
    print(f'  [Plot] ROC curves → {save_path}')


def plot_confusion_matrix(y_true, y_pred, label_names, save_path):
    cm     = confusion_matrix(y_true, y_pred)
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
    fig, ax = plt.subplots(figsize=(8, 7))
    fig.patch.set_facecolor('#0d1117')
    ax.set_facecolor('#0d1117')
    im   = ax.imshow(cm_pct, cmap='Blues')
    cbar = plt.colorbar(im, ax=ax)
    cbar.ax.tick_params(colors='white')
    ax.set_xticks(range(len(label_names)))
    ax.set_yticks(range(len(label_names)))
    ax.set_xticklabels(label_names, rotation=45, ha='right', color='white', fontsize=10)
    ax.set_yticklabels(label_names, color='white', fontsize=10)
    for i in range(len(label_names)):
        for j in range(len(label_names)):
            v = cm_pct[i, j]
            ax.text(j, i, f'{v:.1f}%', ha='center', va='center',
                    color='white' if v < 50 else '#111', fontsize=9)
    ax.set_xlabel('Predicted', color='white', fontsize=12)
    ax.set_ylabel('True',      color='white', fontsize=12)
    ax.set_title('V15.0 — Confusion Matrix (%)', color='white', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='#0d1117')
    plt.close()
    print(f'  [Plot] Confusion matrix → {save_path}')


# ═══════════════════════════════════════════════════════════════
# SAVE MODEL v15
# ═══════════════════════════════════════════════════════════════
def save_artifacts(model: IDSModel, splits: Dict, thresholds: Dict,
                    history: List[Dict], centroids, save_dir: str,
                    knn: Optional[NearestNeighbors] = None) -> Tuple[str, str]:
    os.makedirs(save_dir, exist_ok=True)

    pth_path = os.path.join(save_dir, 'ids_v15_model.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'n_features':       splits['n_features'],
        'n_classes':        splits['n_classes'],
        'hidden':           model.hidden_dim,       # [FIX-3] dùng attribute trực tiếp
        'ae_hidden':        model.ae_hidden_dim,    # [FIX-3]
        'latent_dim':       model.vae.latent_dim,   # [NEW]
        'label_classes':    list(splits['label_encoder'].classes_),
        'known_cats':       splits['known_cats'],
        'zd_cats':          splits['zd_cats'],
        'feat_cols':        splits['feat_cols'],
        'feature_names':    splits.get('feature_names', splits['feat_cols']),
        'categorical_maps': splits.get('categorical_maps', {}),
        'thresholds':       thresholds,
        'version':          'v15.0',
    }, pth_path)
    print(f'  Model weights → {pth_path}')

    # [NEW-5] Lưu history JSON đầy đủ
    hist_path = os.path.join(save_dir, 'ids_v15_history.json')
    with open(hist_path, 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=2, default=str)
    print(f'  History JSON  → {hist_path}')

    pkl_path = os.path.join(save_dir, 'ids_v15_pipeline.pkl')
    pipeline = {
        'scaler':        splits['scaler'],
        'label_encoder': splits['label_encoder'],
        'feat_cols':     splits['feat_cols'],
        'feature_names': splits.get('feature_names', splits['feat_cols']),  # [FIX-4]
        'known_cats':    splits['known_cats'],
        'zd_cats':       splits['zd_cats'],
        'thresholds':    thresholds,
        'categorical_maps': splits.get('categorical_maps', {}),
        'centroids_np':  centroids.cpu().numpy(),
        'n_features':    splits['n_features'],
        'n_classes':     splits['n_classes'],
        'knn':           knn,   # [NEW] lưu kNN detector
        'version':       'v15.0',
    }
    with open(pkl_path, 'wb') as f:
        pickle.dump(pipeline, f)
    print(f'  Pipeline pkl  → {pkl_path}')
    return pth_path, pkl_path


# ═══════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════
def run_full(args: CFG):
    print('\n' + '=' * 70)
    print('IDS v15.0 — VAE + Attention + OOD Ensemble  |  UNSW-NB15')
    print('=' * 70)

    seed_everything(args.seed)

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.plot_dir, exist_ok=True)

    print('\n[1/9] Loading data...')
    df = load_unsw_csvs(args.data_dir)
    df = clean_df(df)

    print('\n[2/9] Preparing splits...')
    splits = prepare_splits(df, seed=args.seed,
                             zd_augment=getattr(args, 'zd_augment_factor', 1))

    le       = splits['label_encoder']
    dos_idx  = int(le.transform(['DoS'])[0]) if 'DoS' in list(le.classes_) else None
    norm_idx = list(le.classes_).index('Normal') if 'Normal' in list(le.classes_) else 0

    print('\n[3/9] Creating loaders...')
    loaders = make_loaders(splits, batch_size=args.batch_size,
                            num_workers=args.num_workers,
                            dos_class_idx=dos_idx, dos_over=5.0,
                            seed=args.seed)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'\n[4/9] Building model (device={device})...')
    model = IDSModel(
        n_features=splits['n_features'], n_classes=splits['n_classes'],
        hidden=args.hidden, ae_hidden=args.ae_hidden, latent_dim=args.latent_dim,
    ).to(device)

    criterion = IDSLoss(
        n_classes=splits['n_classes'], lambda_con=args.lambda_con,
        focal_gamma=args.focal_gamma, dos_class_idx=dos_idx,
        dos_weight=args.dos_weight, kl_weight=args.kl_weight, device=device,
    )

    print('\n[5/9] Training...')
    model, history = train(model, loaders, args, criterion, device, normal_idx=norm_idx)

    print('\n[6/9] Building centroids + KNN detector...')
    centroids = build_centroids(model, splits['X_train'], splits['y_train'],
                                 n_clusters=args.n_clusters, device=device,
                                 seed=args.seed)
    knn = build_knn_detector(model, splits['X_train'],
                              n_neighbors=args.n_neighbors_knn, device=device)

    print('\n[6b/9] Calibrating thresholds on val (KNOWN only)...')
    thresholds = calibrate(model, splits['X_val'], splits['y_val'],
                            args.target_fpr, device, centroids, knn=knn)

    # RE threshold cho decision space plot
    re_val_list = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(splits['X_val']), 512):
            x = torch.FloatTensor(splits['X_val'][i:i+512]).to(device)
            re_val_list.append(model.vae.recon_error(x).cpu().numpy())
    re_thr = float(np.quantile(np.concatenate(re_val_list), 1 - args.target_fpr))

    print('\n[7/9] Evaluating...')
    label_names = list(le.classes_)
    clf_res     = evaluate_classifier(model, splits['X_test'], splits['y_test'],
                                       label_names, device)
    clf_res['y_test'] = splits['y_test']

    zd_res = evaluate_zero_day(
        model, splits['X_test'], splits['y_test'],
        splits['X_zd'],          splits['y_zd'],
        thresholds, centroids, device, knn=knn,
    )

    print('\n[8/9] Saving artifacts...')
    pth_p, pkl_p = save_artifacts(model, splits, thresholds, history, centroids,
                                   args.save_dir, knn=knn)

    print('\n[9/9] Plotting...')
    plot_training_curve(history, os.path.join(args.plot_dir, 'v15_training_curve.png'))
    plot_soc_decision_space(model, splits['X_test'], splits['y_test'],
        splits['X_zd'], 0.5, re_thr, device,
        os.path.join(args.plot_dir, 'v15_decision_space.png'), label_names,
        seed=args.seed)

    per_cls_zd   = zd_res.get('_per_class', {})
    zd_cls_order = sorted(per_cls_zd.keys())
    plot_per_class_proper(label_names, splits['y_test'], clf_res['preds'],
        per_cls_zd, zd_cls_order,
        os.path.join(args.plot_dir, 'v15_per_class_detection.png'), normal_idx=norm_idx)
    plot_roc_curves(zd_res, os.path.join(args.plot_dir, 'v15_roc_curves.png'))
    plot_confusion_matrix(splits['y_test'], clf_res['preds'], label_names,
                          os.path.join(args.plot_dir, 'v15_confusion_matrix.png'))

    best_zd_auc  = max((v['auc']  for k, v in zd_res.items()
                        if not k.startswith('_') and 'auc'  in v), default=0.)
    best_zd_aupr = max((v.get('auprc', 0.) for k, v in zd_res.items()
                        if not k.startswith('_')), default=0.)

    # [NEW-5] Lưu summary JSON
    summary = {
        'version': 'v15.0',
        'known_auc':   clf_res['auc'],
        'known_auprc': clf_res.get('auprc', 0.),
        'best_zd_auc': best_zd_auc,
        'best_zd_aupr': best_zd_aupr,
        'n_features': splits['n_features'],
        'n_classes':  splits['n_classes'],
        'n_epochs':   len(history),
    }
    results_dir = os.path.abspath(os.path.join(args.save_dir, '..', 'results'))
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, 'ids_v15_results.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    print(f'\n{"="*70}')
    print('FINAL SUMMARY - IDS v15.0')
    print(f'{"="*70}')
    print(f'  Known AUC    : {clf_res["auc"]:.4f}')
    print(f'  Known AUPRC  : {clf_res.get("auprc", 0.):.4f}')
    print(f'  Best ZD AUC  : {best_zd_auc:.4f}')
    print(f'  Best ZD AUPR : {best_zd_aupr:.4f}')
    print(f'  Model .pth   : {pth_p}')
    print(f'  Pipeline     : {pkl_p}')
    print(f'{"="*70}')

    return model, zd_res, history


# ═══════════════════════════════════════════════════════════════
# DEMO MODE
# ═══════════════════════════════════════════════════════════════
def run_demo(args: CFG):
    print('\n' + '=' * 60)
    print('DEMO MODE — Synthetic UNSW-NB15-like data')
    print('=' * 60)
    seed_everything(getattr(args, 'seed', 42))
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.plot_dir, exist_ok=True)

    n_feat = 55; n_cls = 5; N = 60000
    X = np.random.randn(N, n_feat).astype(np.float32)
    y = np.random.randint(0, n_cls, N)
    for c in range(n_cls):
        X[y == c] += c * 1.5
    dos_idx_demo = 1
    X[y == dos_idx_demo] = np.random.randn((y == dos_idx_demo).sum(), n_feat) * 0.8

    N_zd = 8000
    X_zd = (np.random.randn(N_zd, n_feat) * 1.5 + 3.).astype(np.float32)
    y_zd = np.array([f'ZD_{i % 5}' for i in range(N_zd)])

    X_tv, X_te, y_tv, y_te = train_test_split(X, y, test_size=0.2, stratify=y)
    X_tr, X_va, y_tr, y_va = train_test_split(X_tv, y_tv, test_size=0.125, stratify=y_tv)

    sc = RobustScaler().fit(X_tr)
    X_tr = sc.transform(X_tr); X_va = sc.transform(X_va)
    X_te = sc.transform(X_te); X_zd = sc.transform(X_zd)

    le = LabelEncoder()
    le.classes_ = np.array([f'Class_{i}' for i in range(n_cls)])

    splits = dict(
        X_train=X_tr, y_train=y_tr, X_val=X_va, y_val=y_va,
        X_test=X_te,  y_test=y_te,  X_zd=X_zd,  y_zd=y_zd,
        n_features=n_feat, n_classes=n_cls,
        label_encoder=le, scaler=sc,
        feat_cols=[f'f{i}' for i in range(n_feat)],
        feature_names=[f'f{i}' for i in range(n_feat)],
        known_cats=[f'Class_{i}' for i in range(n_cls)],
        zd_cats=[f'ZD_{i}' for i in range(5)],
    )

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model  = IDSModel(n_features=n_feat, n_classes=n_cls,
                      hidden=128, ae_hidden=64, latent_dim=16).to(device)

    args.epochs   = min(getattr(args, 'epochs', 10), 10)
    args.patience = min(getattr(args, 'patience', 5), 5)

    criterion = IDSLoss(n_classes=n_cls, lambda_con=0.3, focal_gamma=2.0,
                        dos_class_idx=dos_idx_demo, dos_weight=5.0,
                        kl_weight=0.001, device=device)
    loaders = make_loaders(splits, batch_size=256, num_workers=0,
                            dos_class_idx=dos_idx_demo,
                            seed=getattr(args, 'seed', 42))
    model, history = train(model, loaders, args, criterion, device)

    centroids  = build_centroids(model, X_tr, y_tr, 10, device,
                                 seed=getattr(args, 'seed', 42))
    knn        = build_knn_detector(model, X_tr, n_neighbors=10, device=device)
    thresholds = calibrate(model, X_va, y_va, args.target_fpr, device, centroids, knn=knn)

    label_names = [f'Class_{i}' for i in range(n_cls)]
    clf_res     = evaluate_classifier(model, X_te, y_te, label_names, device)
    clf_res['y_test'] = y_te
    zd_res      = evaluate_zero_day(model, X_te, y_te, X_zd, y_zd,
                                     thresholds, centroids, device, knn=knn)

    pth_p, pkl_p = save_artifacts(model, splits, thresholds, history, centroids,
                                   args.save_dir, knn=knn)

    re_vals = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(X_va), 512):
            x = torch.FloatTensor(X_va[i:i+512]).to(device)
            re_vals.append(model.vae.recon_error(x).cpu().numpy())
    re_thr_demo = float(np.quantile(np.concatenate(re_vals), 0.95))

    plot_training_curve(history, os.path.join(args.plot_dir, 'v15_training_curve.png'))
    plot_soc_decision_space(model, X_te, y_te, X_zd, 0.5, re_thr_demo, device,
                             os.path.join(args.plot_dir, 'v15_decision_space.png'), label_names,
                             seed=getattr(args, 'seed', 42))
    per_cls_zd   = zd_res.get('_per_class', {})
    zd_cls_order = sorted(per_cls_zd.keys())
    plot_per_class_proper(label_names, y_te, clf_res['preds'], per_cls_zd, zd_cls_order,
                           os.path.join(args.plot_dir, 'v15_per_class_detection.png'), normal_idx=0)
    plot_roc_curves(zd_res, os.path.join(args.plot_dir, 'v15_roc_curves.png'))
    plot_confusion_matrix(y_te, clf_res['preds'], label_names,
                          os.path.join(args.plot_dir, 'v15_confusion_matrix.png'))

    print(f'\nDemo done! Plots → {args.plot_dir}')
    return model, zd_res, history


# ═══════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════
if __name__ == '__main__':
    cfg = get_config()
    if cfg.demo:
        run_demo(cfg)
    else:
        run_full(cfg)
