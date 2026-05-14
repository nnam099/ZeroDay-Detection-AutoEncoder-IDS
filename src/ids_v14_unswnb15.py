"""
ids_v14_unswnb15.py — IDS v14.0  Hybrid Anomaly + GradBP
=========================================================
Dataset : UNSW-NB15 (Kaggle)

CẢI TIẾN SO VỚI v13:
  [FIX-DoS]   DoS F1=0.50 → thêm DoS-specific features + FocalLoss per-class weight
  [NEW]       Hybrid detector: Supervised f(x) + AutoEncoder g(x) → SOC Decision Space
  [NEW]       Plots giống v5: SOC Decision Space scatter + Per-class detection bar chart
  [NEW]       Save model  → .pth  (weights) + .pkl (scaler / label_encoder / feat_cols)
  [NEW]       Contrastive prototype head giúp DoS tách xa Normal
  [TUNE]      WeightedRandomSampler tăng DoS weight x5
  [TUNE]      AE reconstruction error làm anomaly score thứ 2
"""

import os, sys, glob, json, copy, time, pickle, warnings, random
from collections import deque
# Fix UnicodeEncodeError khi print duong dan tieng Viet tren Windows terminal
if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    try: sys.stdout.reconfigure(encoding='utf-8')
    except Exception: pass
if sys.stderr and hasattr(sys.stderr, 'reconfigure'):
    try: sys.stderr.reconfigure(encoding='utf-8')
    except Exception: pass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')           # không cần GUI
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
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
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings('ignore')
torch.backends.cudnn.benchmark = True

# ═══════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════
class CFG:
    data_dir    = '/kaggle/input'
    save_dir    = '/kaggle/working/checkpoints_v14'
    plot_dir    = '/kaggle/working/plots_v14'
    demo        = False

    # Training
    epochs      = 100
    batch_size  = 512
    lr          = 3e-4
    hidden      = 256
    ae_hidden   = 128
    patience    = 20

    # Loss
    lambda_con  = 0.3
    focal_gamma = 2.0
    dos_weight  = 3.0
    recon_dos_penalty = 2.0

    # Zero-day
    target_fpr        = 0.05
    adaptive_threshold = False
    n_clusters        = 25
    zd_augment_factor = 1

    num_workers = 2
    seed        = 42


def resolve_paths(cfg):
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


def get_config():
    in_nb = False
    try:
        shell = get_ipython().__class__.__name__   # noqa
        in_nb = shell in ('ZMQInteractiveShell', 'Shell')
    except NameError:
        pass
    if in_nb:
        return resolve_paths(CFG)
    import argparse
    p = argparse.ArgumentParser()
    for k, v in vars(CFG).items():
        if k.startswith('_'):
            continue
        if isinstance(v, bool):
            p.add_argument(f'--{k}', action='store_true', default=v)
        elif v is not None:
            p.add_argument(f'--{k}', type=type(v), default=v)
    args = p.parse_args()
    for k, v in vars(args).items():
        setattr(CFG, k, v)
    return resolve_paths(CFG)


# ═══════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════
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
# MODEL
# ═══════════════════════════════════════════════════════════════
class ResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim,dim), nn.LayerNorm(dim), nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(dim,dim), nn.LayerNorm(dim),
        )
    def forward(self,x): return F.gelu(x + self.net(x))


class IDSBackbone(nn.Module):
    def __init__(self, n_features, hidden=256):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(n_features, hidden), nn.LayerNorm(hidden), nn.GELU(),
        )
        self.res1 = ResBlock(hidden)
        self.res2 = ResBlock(hidden)
        self.res3 = ResBlock(hidden)
        self.hidden = hidden

    def forward(self, x):
        h = self.input_proj(x)
        h = self.res1(h)
        h = self.res2(h)
        h = self.res3(h)
        return h


class AutoEncoder(nn.Module):
    def __init__(self, n_features, ae_hidden=128):
        super().__init__()
        mid = ae_hidden
        self.enc = nn.Sequential(
            nn.Linear(n_features, mid*2), nn.GELU(),
            nn.Linear(mid*2, mid),        nn.GELU(),
            nn.Linear(mid, mid//2),
        )
        self.dec = nn.Sequential(
            nn.Linear(mid//2, mid),       nn.GELU(),
            nn.Linear(mid, mid*2),        nn.GELU(),
            nn.Linear(mid*2, n_features),
        )

    def forward(self, x):
        z    = self.enc(x)
        x_hat= self.dec(z)
        return x_hat

    def recon_error(self, x):
        return (self.forward(x) - x).pow(2).mean(dim=-1)


class IDSModel(nn.Module):
    def __init__(self, n_features, n_classes, hidden=256, ae_hidden=128):
        super().__init__()
        self.n_classes   = n_classes
        self.backbone    = IDSBackbone(n_features, hidden)
        self.classifier  = nn.Linear(hidden, n_classes)
        self.proj_head   = nn.Sequential(
            nn.Linear(hidden,128), nn.GELU(), nn.Linear(128,64)
        )
        self.ae          = AutoEncoder(n_features, ae_hidden)
        self.log_temp    = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        fv     = self.backbone(x)
        logits = self.classifier(fv)
        return logits, fv

    def get_embed(self, x):
        fv = self.backbone(x)
        return F.normalize(self.proj_head(fv), dim=-1)

    def attack_prob(self, x):
        logits, _ = self.forward(x)
        probs  = torch.softmax(logits, dim=-1)
        return probs

    def energy_score(self, x):
        logits, _ = self.forward(x)
        T = self.log_temp.exp().clamp(0.5,5.0)
        return -T * torch.logsumexp(logits/T, dim=-1)

    def gradbp_score(self, x):
        x  = x.detach()
        fv = self.backbone(x)
        fv.requires_grad_(True)
        logits = self.classifier(fv)
        pred   = logits.argmax(dim=-1)
        loss   = F.cross_entropy(logits, pred)
        grad   = torch.autograd.grad(loss, fv)[0]
        return grad.norm(dim=-1)

    def fv_cluster_score(self, x, centroids):
        _, fv     = self.forward(x)
        fv_n      = F.normalize(fv, dim=-1)
        cent_n    = F.normalize(centroids, dim=-1)
        return torch.cdist(fv_n, cent_n).min(dim=-1).values

    def hybrid_score(self, x, centroids=None, hybrid_meta=None):
        if hybrid_meta is None:
            raise ValueError('hybrid_meta is required for learned hybrid scoring')
        re  = self.ae.recon_error(x)
        probs = torch.softmax(self.forward(x)[0], dim=-1)
        cls_s = 1 - probs.max(dim=-1).values
        coef = torch.tensor(hybrid_meta['coef'], dtype=re.dtype, device=re.device)
        intercept = torch.tensor(float(hybrid_meta.get('intercept', 0.0)),
                                 dtype=re.dtype, device=re.device)
        return torch.sigmoid(intercept + coef[0] * re + coef[1] * cls_s)


# ═══════════════════════════════════════════════════════════════
# LOSS
# ═══════════════════════════════════════════════════════════════
class FocalLoss(nn.Module):
    def __init__(self, n_classes, gamma=2.5, label_smooth=0.05, class_weights=None,
                 dos_class_idx=None, recon_class_idx=None, recon_dos_penalty=2.0):
        super().__init__()
        self.gamma  = gamma
        self.ls     = label_smooth
        self.nc     = n_classes
        self.w      = class_weights
        self.dos_idx = dos_class_idx
        self.recon_idx = recon_class_idx
        self.recon_dos_penalty = float(recon_dos_penalty)

    def forward(self, logits, labels):
        ce   = F.cross_entropy(logits, labels, reduction='none',
                               label_smoothing=self.ls, weight=self.w)
        pt   = torch.exp(-ce)
        loss = (1-pt)**self.gamma * ce
        if self.dos_idx is not None and self.recon_idx is not None:
            preds = logits.argmax(dim=1)
            recon_as_dos = (labels == self.recon_idx) & (preds == self.dos_idx)
            dos_as_recon = (labels == self.dos_idx) & (preds == self.recon_idx)
            penalty_mask = recon_as_dos | dos_as_recon
            if penalty_mask.any():
                mult = torch.ones_like(loss)
                mult[penalty_mask] = self.recon_dos_penalty
                loss = loss * mult
        return loss.mean()


class SupConLoss(nn.Module):
    def __init__(self, T=0.15, dos_class_idx=None, recon_class_idx=None,
                 hard_negative_weight=0.20, hard_negative_topk=64, hard_negative_margin=0.20):
        super().__init__()
        self.T = T
        self.dos_idx = dos_class_idx
        self.recon_idx = recon_class_idx
        self.hn_weight = float(hard_negative_weight)
        self.hn_topk = int(hard_negative_topk)
        self.hn_margin = float(hard_negative_margin)

    def forward(self, feats, labels):
        B    = feats.shape[0]
        sim  = torch.matmul(feats, feats.T) / self.T
        sim  = sim - sim.detach().max()
        lmat = labels.view(-1,1)
        pos  = (lmat == lmat.T).float().fill_diagonal_(0)
        eye  = torch.eye(B, device=feats.device, dtype=torch.bool)
        denom= (torch.exp(sim) * (~eye).float()).sum(1, keepdim=True).clamp(min=1e-8)
        lp   = sim - torch.log(denom)
        cnt  = pos.sum(1).clamp(min=1e-8)
        loss = -(pos*lp).sum(1)/cnt
        valid= pos.sum(1) > 0
        hard_neg = self._recon_dos_hard_negative_loss(feats, labels)
        if not valid.any():
            return hard_neg
        return loss[valid].mean() + hard_neg

    def _recon_dos_hard_negative_loss(self, feats, labels):
        if self.dos_idx is None or self.recon_idx is None or self.hn_weight <= 0:
            return torch.tensor(0.0, device=feats.device, requires_grad=True)
        lmat = labels.view(-1, 1)
        recon_dos = ((lmat == self.recon_idx) & (lmat.T == self.dos_idx)) | \
                    ((lmat == self.dos_idx) & (lmat.T == self.recon_idx))
        if not recon_dos.any():
            return torch.tensor(0.0, device=feats.device, requires_grad=True)
        cos = torch.matmul(feats, feats.T)
        hard_scores = cos[recon_dos]
        if hard_scores.numel() > self.hn_topk:
            hard_scores = torch.topk(hard_scores, self.hn_topk).values
        penalty = F.softplus((hard_scores - self.hn_margin) / self.T).mean()
        return self.hn_weight * penalty


class IDSLoss(nn.Module):
    def __init__(self, n_classes, lambda_con=0.3, focal_gamma=2.0,
                 dos_class_idx=None, dos_weight=5.0, recon_class_idx=None,
                 recon_dos_penalty=2.0, n_features=None, device='cpu'):
        super().__init__()
        w = torch.ones(n_classes, device=device)
        if dos_class_idx is not None:
            w[dos_class_idx] = dos_weight
        w = w / w.mean()
        self.focal = FocalLoss(n_classes, gamma=focal_gamma,
                               label_smooth=0.05, class_weights=w.to(device),
                               dos_class_idx=dos_class_idx,
                               recon_class_idx=recon_class_idx,
                               recon_dos_penalty=recon_dos_penalty)
        self.con   = SupConLoss(T=0.15, dos_class_idx=dos_class_idx,
                                recon_class_idx=recon_class_idx)
        self.lam   = lambda_con
        self.ae_loss_fn = nn.MSELoss()

    def forward(self, logits, labels, embeds, x, x_hat):
        lf  = self.focal(logits, labels)
        lc  = self.con(embeds, labels)
        lae = self.ae_loss_fn(x_hat, x)
        lf  = torch.nan_to_num(lf,  nan=0.0, posinf=10.0)
        lc  = torch.nan_to_num(lc,  nan=0.0, posinf=10.0)
        lae = torch.nan_to_num(lae, nan=0.0, posinf=10.0)
        total = lf + self.lam*lc + 0.5*lae
        return {'total':total, 'focal':lf, 'con':lc, 'ae':lae}


# ═══════════════════════════════════════════════════════════════
# DATASET / LOADER
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
# TRAINING
# ═══════════════════════════════════════════════════════════════
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    tot_loss = ncorr = ntot = 0
    use_amp = (device == 'cuda')
    scaler  = torch.amp.GradScaler('cuda') if use_amp else None

    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)

        if use_amp:
            with torch.amp.autocast('cuda'):
                logits, _ = model(X)
                embeds    = model.get_embed(X)
                x_hat     = model.ae(X)
                losses    = criterion(logits, y, embeds, X, x_hat)
                loss      = losses['total']
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits, _ = model(X)
            embeds    = model.get_embed(X)
            x_hat     = model.ae(X)
            losses    = criterion(logits, y, embeds, X, x_hat)
            loss      = losses['total']
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        loss_val = loss.item()
        if not np.isfinite(loss_val):
            continue
        tot_loss += loss_val * len(y)
        ncorr    += (logits.argmax(1)==y).sum().item()
        ntot     += len(y)

    if ntot == 0:
        return {'loss': float('nan'), 'acc': 0.0}
    return {'loss': tot_loss/ntot, 'acc': ncorr/ntot}


@torch.no_grad()
def eval_epoch(model, loader, device):
    model.eval()
    all_probs, all_labels = [], []
    for X, y in loader:
        logits, _ = model(X.to(device))
        all_probs.append(torch.softmax(logits,dim=-1).cpu().numpy())
        all_labels.append(y.numpy())
    probs  = np.concatenate(all_probs)
    labels = np.concatenate(all_labels)
    binary = (labels != 0).astype(int)
    score  = 1 - probs[:,0]
    try:    auc = roc_auc_score(binary, score)
    except: auc = 0.5
    return {'auc':auc, 'acc':(probs.argmax(1)==labels).mean()}


@torch.no_grad()
def _collect_loader_predictions(model, loader, device):
    model.eval()
    preds, labels = [], []
    for X, y in loader:
        logits, _ = model(X.to(device))
        preds.append(logits.argmax(1).cpu().numpy())
        labels.append(y.numpy())
    return np.concatenate(labels), np.concatenate(preds)


def _format_class_name(label_names, idx):
    if label_names and idx < len(label_names):
        return str(label_names[idx])
    return f'Class_{idx}'


def log_top_confusions(model, loader, device, label_names=None, epoch=None, top_k=2):
    labels, preds = _collect_loader_predictions(model, loader, device)
    n_classes = len(label_names) if label_names else int(max(labels.max(), preds.max()) + 1)
    cm = confusion_matrix(labels, preds, labels=list(range(n_classes)))
    rows = []
    for true_idx in range(n_classes):
        row_total = int(cm[true_idx].sum())
        if row_total == 0:
            continue
        for pred_idx in range(n_classes):
            if pred_idx == true_idx:
                continue
            count = int(cm[true_idx, pred_idx])
            if count <= 0:
                continue
            rows.append({
                'true': _format_class_name(label_names, true_idx),
                'pred': _format_class_name(label_names, pred_idx),
                'count': count,
                'row_pct': 100.0 * count / row_total,
            })
    rows = sorted(rows, key=lambda r: (r['count'], r['row_pct']), reverse=True)[:top_k]
    title = 'Top validation confusions'
    if epoch is not None:
        title += f' @ epoch {epoch}'
    print(f'\n  {title}')
    print(f'  {"True":<18} {"Pred":<18} {"Count":>7} {"Row%":>8}')
    print(f'  {"-"*55}')
    if not rows:
        print(f'  {"<none>":<18} {"<none>":<18} {0:>7} {0.0:>7.1f}%')
    for item in rows:
        print(f'  {item["true"]:<18} {item["pred"]:<18} {item["count"]:>7} {item["row_pct"]:>7.1f}%')
    return rows


def train(model, loaders, args, criterion, device, label_names=None):
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    def warmup_cos(ep):
        warmup=5
        if ep<warmup: return ep/warmup
        prog=(ep-warmup)/max(args.epochs-warmup,1)
        return 0.5*(1+np.cos(np.pi*prog))
    sched    = torch.optim.lr_scheduler.LambdaLR(optimizer, warmup_cos)
    best_auc = 0.; best_sd = None; patience = 0; history = []

    print(f'\n{"="*65}')
    print(f'Training v14 | epochs={args.epochs} lr={args.lr} hidden={args.hidden}')
    print(f'{"="*65}')

    for ep in range(1, args.epochs+1):
        t0  = time.time()
        trm = train_epoch(model, loaders['train'], optimizer, criterion, device)
        vam = eval_epoch(model,  loaders['val'],   device)
        log_top_confusions(model, loaders['val'], device, label_names=label_names, epoch=ep, top_k=2)
        sched.step()
        is_best = vam['auc'] > best_auc
        if is_best:
            best_auc = vam['auc']
            best_sd  = copy.deepcopy(model.state_dict())
            patience = 0
        else:
            patience += 1
        history.append({**trm, **{f'val_{k}':v for k,v in vam.items()}})
        star = '*' if is_best else ' '
        print(f'{star}[{ep:3d}/{args.epochs}] loss={trm["loss"]:.4f} acc={trm["acc"]:.4f}'
              f'  vAUC={vam["auc"]:.4f}  P={patience}  ({time.time()-t0:.1f}s)')
        if patience >= args.patience:
            print(f'  Early stop at ep {ep}')
            break

    if best_sd: model.load_state_dict(best_sd)
    print(f'\nBest Val AUC = {best_auc:.4f}')
    return model, history


# ═══════════════════════════════════════════════════════════════
# SCORING HELPERS
# ═══════════════════════════════════════════════════════════════
class AdaptiveThreshold:
    def __init__(self, window_size=1000, target_fpr=0.05):
        self.window_size = int(window_size)
        self.target_fpr = float(target_fpr)
        self.buffer = deque(maxlen=self.window_size)
        self.threshold = float('inf')

    def update(self, re_scores: np.ndarray):
        scores = np.asarray(re_scores, dtype=np.float64).reshape(-1)
        scores = scores[np.isfinite(scores)]
        self.buffer.extend(float(score) for score in scores)
        if self.buffer:
            self.threshold = float(np.quantile(
                np.asarray(self.buffer, dtype=np.float64),
                1.0 - self.target_fpr,
            ))
        return self.threshold

    def __call__(self, re_score: float) -> bool:
        return bool(float(re_score) > self.threshold)


def _sigmoid_np(values):
    values = np.clip(values, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-values))


def compute_hybrid_meta_score(ae_re, softmax_score, hybrid_meta):
    if not hybrid_meta:
        raise ValueError('hybrid_meta is required for learned hybrid scoring')
    coef = np.asarray(hybrid_meta.get('coef'), dtype=np.float64).reshape(-1)
    if coef.size < 2:
        raise ValueError('hybrid_meta must contain two coefficients: ae_re and softmax')
    intercept = float(hybrid_meta.get('intercept', 0.0))
    ae_re = np.asarray(ae_re, dtype=np.float64)
    softmax_score = np.asarray(softmax_score, dtype=np.float64)
    return _sigmoid_np(intercept + coef[0] * ae_re + coef[1] * softmax_score).astype(np.float32)


def _hybrid_base_features(model, X, device, batch=512):
    model.eval()
    features = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            x = torch.FloatTensor(X[i:i+batch]).to(device)
            probs = torch.softmax(model.forward(x)[0], dim=-1)
            softmax_score = (1.0 - probs.max(dim=-1).values).cpu().numpy()
            ae_re = model.ae.recon_error(x).cpu().numpy()
            features.append(np.column_stack([ae_re, softmax_score]))
    if not features:
        return np.empty((0, 2), dtype=np.float32)
    return np.concatenate(features, axis=0).astype(np.float32)


def fit_hybrid_meta_learner(model, X_val_known, X_zd, device, seed=42):
    known_features = _hybrid_base_features(model, X_val_known, device)
    zd_features = _hybrid_base_features(model, X_zd, device)
    X_meta = np.vstack([known_features, zd_features])
    y_meta = np.concatenate([
        np.zeros(len(known_features), dtype=np.int64),
        np.ones(len(zd_features), dtype=np.int64),
    ])
    if X_meta.size == 0 or len(np.unique(y_meta)) < 2:
        raise ValueError('hybrid meta-learner requires known validation and zero-day samples')

    learner = LogisticRegression(
        max_iter=1000,
        class_weight='balanced',
        solver='lbfgs',
        random_state=seed,
    )
    learner.fit(X_meta, y_meta)
    hybrid_meta = {
        'type': 'logistic_regression',
        'features': ['ae_re', 'softmax'],
        'coef': [float(learner.coef_[0, 0]), float(learner.coef_[0, 1])],
        'intercept': float(learner.intercept_[0]),
        'train_rows': int(len(X_meta)),
        'positive_rows': int(y_meta.sum()),
    }
    print('\n  Hybrid meta-learner weights')
    print(f'    {"feature":<16} {"coef":>12}')
    print(f'    {"-"*29}')
    print(f'    {"ae_re":<16} {hybrid_meta["coef"][0]:>12.6f}')
    print(f'    {"1-max_prob":<16} {hybrid_meta["coef"][1]:>12.6f}')
    print(f'    {"intercept":<16} {hybrid_meta["intercept"]:>12.6f}')
    return hybrid_meta


def _batch_scores(model, X, device, centroids=None, hybrid_meta=None, batch=512):
    model.eval()
    s_sm, s_fvc, s_re, s_hyb = [], [], [], []
    with torch.no_grad():
        for i in range(0,len(X),batch):
            x = torch.FloatTensor(X[i:i+batch]).to(device)
            probs = torch.softmax(model.forward(x)[0], dim=-1)
            softmax_np = (1-probs.max(dim=-1).values).cpu().numpy()
            s_sm.append(softmax_np)
            re = model.ae.recon_error(x)
            re_np = re.cpu().numpy()
            s_re.append(re_np)
            if centroids is not None:
                s_fvc.append(model.fv_cluster_score(x,centroids).cpu().numpy())
            if hybrid_meta is not None:
                s_hyb.append(compute_hybrid_meta_score(re_np, softmax_np, hybrid_meta))
    out = {
        'softmax': np.concatenate(s_sm),
        'ae_re':   np.concatenate(s_re),
    }
    if centroids is not None:
        out['fv_cluster'] = np.concatenate(s_fvc)
    if hybrid_meta is not None:
        out['hybrid']     = np.concatenate(s_hyb)
    return out


def _batch_gradbp(model, X, device, batch=512):
    scores = []
    for i in range(0,len(X),batch):
        x = torch.FloatTensor(X[i:i+batch]).to(device)
        scores.append(model.gradbp_score(x).detach().cpu().numpy())
    return np.concatenate(scores)


def build_centroids(model, X_tr, y_tr, n_clusters=25, device='cpu', seed=42):
    model.eval()
    fvs = []
    with torch.no_grad():
        for i in range(0, len(X_tr), 1024):
            x = torch.FloatTensor(X_tr[i:i+1024]).to(device)
            _, fv = model(x)
            fv_np = F.normalize(fv, dim=-1).cpu().float().numpy()
            fvs.append(fv_np)
    all_fvs = np.concatenate(fvs)
    all_fvs = np.nan_to_num(all_fvs, nan=0.0, posinf=0.0, neginf=0.0)

    centers = []
    for cls in range(len(np.unique(y_tr))):
        m  = y_tr == cls
        cv = all_fvs[m]
        if len(cv) == 0:
            continue
        cv = cv[np.isfinite(cv).all(axis=1)]
        if len(cv) == 0:
            continue
        k = min(n_clusters, len(cv))
        if k == 1:
            centers.append(cv.mean(0, keepdims=True))
        else:
            km = MiniBatchKMeans(n_clusters=k, random_state=seed, n_init=3, batch_size=2048)
            centers.append(km.fit(cv).cluster_centers_)
    c = np.concatenate(centers)
    print(f'  Centroids: {len(c)}')
    return torch.FloatTensor(c).to(device)


@torch.no_grad()
def class_prototype_cosine_similarity(model, X, y, class_a, class_b, device='cpu', batch=512):
    model.eval()
    emb, labels = [], []
    y = np.asarray(y)
    for i in range(0, len(X), batch):
        x = torch.FloatTensor(X[i:i+batch]).to(device)
        emb.append(model.get_embed(x).detach().cpu())
        labels.append(torch.LongTensor(y[i:i+batch]))
    emb = torch.cat(emb, dim=0)
    labels = torch.cat(labels, dim=0)
    mask_a = labels == int(class_a)
    mask_b = labels == int(class_b)
    if not mask_a.any() or not mask_b.any():
        raise ValueError('Both classes must have at least one sample')
    proto_a = F.normalize(emb[mask_a].mean(dim=0), dim=0)
    proto_b = F.normalize(emb[mask_b].mean(dim=0), dim=0)
    return float(torch.dot(proto_a, proto_b).item())


def calibrate(model, X_val, y_val, target_fpr, device, centroids, hybrid_meta=None):
    scores = _batch_scores(model, X_val, device, centroids, hybrid_meta=hybrid_meta)
    scores['gradbp_l2'] = _batch_gradbp(model, X_val, device)
    thr = {}
    print(f'\n  Thresholds @ FPR={target_fpr*100:.0f}%')
    for m, arr in scores.items():
        t = float(np.quantile(arr, 1.0-target_fpr))
        thr[m] = t
        print(f'    {m:<16} thr={t:.6f}  actual_FPR={(arr>t).mean():.4f}')
    return thr


# ═══════════════════════════════════════════════════════════════
# EVALUATION
# ═══════════════════════════════════════════════════════════════
def compute_adaptive_threshold_trace(model, X_test, y_test, normal_idx,
                                     seed_re_scores, target_fpr, device,
                                     window_size=1000):
    ae_scores = _batch_scores(model, X_test, device)['ae_re']
    tracker = AdaptiveThreshold(window_size=window_size, target_fpr=target_fpr)
    tracker.update(seed_re_scores)
    thresholds, decisions = [], []
    for re_score, label in zip(ae_scores, y_test):
        decisions.append(tracker(float(re_score)))
        if int(label) == int(normal_idx):
            tracker.update(np.asarray([re_score], dtype=np.float32))
        thresholds.append(tracker.threshold)
    return {
        'ae_scores': ae_scores,
        'thresholds': np.asarray(thresholds, dtype=np.float32),
        'decisions': np.asarray(decisions, dtype=bool),
        'final_threshold': float(tracker.threshold),
    }


def evaluate_classifier(model, X_te, y_te, label_names, device):
    model.eval()
    preds, probs_list = [], []
    with torch.no_grad():
        for i in range(0,len(X_te),512):
            x = torch.FloatTensor(X_te[i:i+512]).to(device)
            lg, _ = model(x)
            preds.append(lg.argmax(1).cpu().numpy())
            probs_list.append(torch.softmax(lg,dim=-1).cpu().numpy())
    preds = np.concatenate(preds)
    probs = np.concatenate(probs_list)
    print(classification_report(y_te, preds, target_names=label_names, digits=4))
    ni = label_names.index('Normal') if 'Normal' in label_names else 0
    bin_ = (y_te!=ni).astype(int)
    score = 1-probs[:,ni]
    try: auc = roc_auc_score(bin_, score)
    except: auc = 0.5
    print(f'  AUC(Normal vs Attack): {auc:.4f}')
    return {'preds':preds,'probs':probs,'auc':auc}


def evaluate_zero_day(model, X_kn, y_kn, X_zd, y_zd, thr, centroids, device, hybrid_meta=None):
    print(f'\n{"="*65}')
    print(f'ZERO-DAY DETECTION  |  Known={len(X_kn):,}  ZD={len(X_zd):,}')
    print(f'{"="*65}')
    sk = _batch_scores(model, X_kn, device, centroids, hybrid_meta=hybrid_meta)
    sz = _batch_scores(model, X_zd, device, centroids, hybrid_meta=hybrid_meta)
    sk['gradbp_l2'] = _batch_gradbp(model, X_kn, device)
    sz['gradbp_l2'] = _batch_gradbp(model, X_zd, device)

    true = np.concatenate([np.zeros(len(X_kn)), np.ones(len(X_zd))])
    results = {}
    print(f'\n  {"Method":<16} {"AUC":>8} {"TPR@1%":>10} {"TPR@5%":>10}')
    print(f'  {"-"*48}')
    # Energy is intentionally excluded from OOD comparison: observed AUC < 0.6,
    # which is not materially better than a random baseline for this task.
    for m in ['gradbp_l2','hybrid','ae_re','softmax','fv_cluster']:
        if m not in sk: continue
        sc_all = np.concatenate([sk[m], sz[m]])
        try: auc = roc_auc_score(true, sc_all)
        except: auc=0.5
        fpr_a, tpr_a, _ = roc_curve(true, sc_all)
        def tpr_at(tfpr):
            idx = np.searchsorted(fpr_a, tfpr)
            return float(tpr_a[min(idx,len(tpr_a)-1)])
        t1,t5 = tpr_at(0.01), tpr_at(0.05)
        print(f'  {m:<16} {auc:>8.4f} {t1:>10.4f} {t5:>10.4f}')
        results[m] = {'auc':auc,'tpr_1':t1,'tpr_5':t5,
                      'fpr':fpr_a.tolist(),'tpr':tpr_a.tolist(),
                      's_known':sk[m],'s_zd':sz[m]}

    best = max(results, key=lambda m: results[m]['auc'])
    bt   = thr.get(best, float(np.quantile(sk[best],0.95)))
    print(f'\n  Per-class recall [{best}@thr={bt:.5f}]:')
    per_cls = {}
    for cls in np.unique(y_zd):
        mask = y_zd==cls
        n    = mask.sum()
        det  = (sz[best][mask]>bt).sum()
        r    = det/n if n>0 else 0.
        per_cls[str(cls)] = {'n':int(n),'recall':r}
        bar  = '#'*int(r*20)+'.'*(20-int(r*20))
        print(f'    {str(cls):<30} [{bar}] {r:.1%}  (n={n:,})')

    results['_per_class']   = per_cls
    results['_best_method'] = best
    results['_scores_known']= sk
    results['_scores_zd']   = sz
    return results


# ═══════════════════════════════════════════════════════════════
# PLOTS
# ═══════════════════════════════════════════════════════════════
def _attack_probs_batch(model, X, device, batch=512):
    model.eval()
    prob_attack = []
    with torch.no_grad():
        for i in range(0,len(X),batch):
            x  = torch.FloatTensor(X[i:i+batch]).to(device)
            lg, _ = model(x)
            pr = torch.softmax(lg,dim=-1)
            prob_attack.append((1-pr[:,0]).cpu().numpy())
    return np.concatenate(prob_attack)


def plot_soc_decision_space(model, X_kn, y_kn_labels, X_zd, p_thr, re_thr,
                             device, save_path, label_names, n_sample=5000,
                             seed=42):
    model.eval()
    rng = np.random.default_rng(seed)

    def sample(X, y, n):
        idx = rng.choice(len(X), min(n,len(X)), replace=False)
        return X[idx], y[idx]

    normal_idx  = label_names.index('Normal') if 'Normal' in label_names else 0
    mask_benign = y_kn_labels == normal_idx
    mask_katk   = ~mask_benign

    Xb, _   = sample(X_kn[mask_benign], y_kn_labels[mask_benign], n_sample)
    Xka, _  = sample(X_kn[mask_katk],  y_kn_labels[mask_katk],   n_sample)
    Xzd, _  = sample(X_zd, np.zeros(len(X_zd)), n_sample)

    def get_xy(X):
        prob = _attack_probs_batch(model, X, device)
        with torch.no_grad():
            re_list=[]
            for i in range(0,len(X),512):
                x=torch.FloatTensor(X[i:i+512]).to(device)
                re_list.append(model.ae.recon_error(x).cpu().numpy())
        re = np.concatenate(re_list)
        return prob, re

    pb, reb    = get_xy(Xb)
    pka, reka  = get_xy(Xka)
    pzd, rezd  = get_xy(Xzd)

    fig, ax = plt.subplots(figsize=(11,8))
    fig.patch.set_facecolor('#0d1117')
    ax.set_facecolor('#0d1117')

    ax.scatter(pb,  reb,  c='#00BFFF', s=8, alpha=0.35, label='Benign',        zorder=2)
    ax.scatter(pka, reka, c='#FF6B6B', s=8, alpha=0.45, label='Known Attack',   zorder=3)
    ax.scatter(pzd, rezd, c='#FFA500', s=8, alpha=0.45, label='Zero-Day Attack',zorder=4)

    ax.axvline(p_thr,  color='white', linestyle='--', lw=1.2, label=f'P-Thr {p_thr:.2f}')
    ax.axhline(re_thr, color='white', linestyle=':',  lw=1.2, label=f'RE-Thr {re_thr:.3f}')

    ax.set_yscale('log')
    ax.set_xlabel('Probability of Attack (Classifier)', color='white', fontsize=12)
    ax.set_ylabel('Reconstruction Error (Anomaly)',     color='white', fontsize=12)
    ax.set_title('V14.0 - SOC Decision Space', color='white', fontsize=14, fontweight='bold')
    ax.tick_params(colors='white')
    for sp in ax.spines.values(): sp.set_edgecolor('#444')
    ax.legend(facecolor='#1a1f2e', edgecolor='#555', labelcolor='white', fontsize=10, loc='lower right')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='#0d1117')
    plt.close()
    print(f'  [Plot] SOC Decision Space -> {save_path}')


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
    zd_recalls = [per_cls_zd[c]['recall']*100 for c in zd_classes_order]

    fig, axes = plt.subplots(1, 2, figsize=(18, max(7, max(len(known_names),len(zd_names))*0.7)))
    fig.patch.set_facecolor('#0d1117')

    def draw_panel(ax, names, vals, color, title):
        ax.set_facecolor('#0d1117')
        bars = ax.barh(names, vals, color=color, height=0.6)
        for bar, val in zip(bars, vals):
            ax.text(min(bar.get_width()+1, 108),
                    bar.get_y()+bar.get_height()/2,
                    f'{val:.1f}%', va='center', color='white', fontsize=10)
        ax.set_xlim(0, 115)
        ax.set_xlabel('Detection Rate (%)', color='white', fontsize=11)
        ax.set_title(title, color='white', fontsize=11)
        ax.tick_params(colors='white')
        for sp in ax.spines.values(): sp.set_edgecolor('#333')

    draw_panel(axes[0], known_names, known_recalls, '#00A8E8',
               'Known Attacks Detection Rate (%)\n(Supervised Head)')
    draw_panel(axes[1], zd_names, zd_recalls, '#FFA500',
               'Zero-Day Attacks Detection Rate (%)\n(Hybrid Anomaly)')

    fig.suptitle('V14.0 - PER-CLASS DETECTION PERFORMANCE',
                 color='white', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='#0d1117')
    plt.close()
    print(f'  [Plot] Per-class -> {save_path}')


def plot_training_curve(history, save_path):
    epochs = list(range(1, len(history)+1))
    losses = [h['loss']    for h in history]
    aucs   = [h['val_auc'] for h in history]
    accs   = [h['val_acc'] for h in history]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.patch.set_facecolor('#0d1117')
    for ax, d, t, c in zip(axes,
                            [losses, aucs, accs],
                            ['Train Loss','Val AUC','Val Accuracy'],
                            ['#FF6B6B','#00BFFF','#FFA500']):
        ax.set_facecolor('#0d1117')
        ax.plot(epochs, d, c=c, lw=2)
        ax.set_title(t, color='white', fontsize=12)
        ax.tick_params(colors='white')
        ax.set_xlabel('Epoch', color='white')
        for sp in ax.spines.values(): sp.set_edgecolor('#333')

    fig.suptitle('V14.0 - Training Curves', color='white', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='#0d1117')
    plt.close()
    print(f'  [Plot] Training curve -> {save_path}')


def plot_threshold_drift(threshold_trace, save_path):
    thresholds = np.asarray(threshold_trace.get('thresholds', []), dtype=np.float32)
    if thresholds.size == 0:
        return
    fig, ax = plt.subplots(figsize=(12, 4))
    fig.patch.set_facecolor('#0d1117')
    ax.set_facecolor('#0d1117')
    ax.plot(np.arange(len(thresholds)), thresholds, color='#FFA500', lw=1.8)
    ax.set_xlabel('Test timeline index', color='white', fontsize=11)
    ax.set_ylabel('AE threshold', color='white', fontsize=11)
    ax.set_title('V14.0 - Adaptive AE Threshold Drift', color='white', fontsize=13, fontweight='bold')
    ax.tick_params(colors='white')
    ax.grid(color='#26313b', linestyle='--', linewidth=0.6, alpha=0.7)
    for sp in ax.spines.values(): sp.set_edgecolor('#333')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='#0d1117')
    plt.close()
    print(f'  [Plot] Threshold drift -> {save_path}')


def plot_roc_curves(zd_results, save_path):
    fig, ax = plt.subplots(figsize=(8, 7))
    fig.patch.set_facecolor('#0d1117')
    ax.set_facecolor('#0d1117')

    method_colors = {
        'gradbp_l2':'#FF6B6B', 'hybrid':'#FFA500', 'ae_re':'#00BFFF',
        'softmax':'#DDA0DD',   'fv_cluster':'#F0E68C',
    }
    for m, r in zd_results.items():
        if m.startswith('_') or 'fpr' not in r: continue
        fpr = np.array(r['fpr']); tpr = np.array(r['tpr'])
        auc = r['auc']
        c   = method_colors.get(m,'white')
        ls  = '--' if m == 'softmax' else '-'
        ax.plot(fpr, tpr, c=c, lw=2, ls=ls, label=f'{m}  AUC={auc:.4f}')

    ax.plot([0,1],[0,1],'--', color='#555', lw=1)
    ax.set_xlabel('False Positive Rate', color='white', fontsize=12)
    ax.set_ylabel('True Positive Rate',  color='white', fontsize=12)
    ax.set_title('V14.0 - Zero-Day Detection ROC', color='white', fontsize=13, fontweight='bold')
    ax.tick_params(colors='white')
    for sp in ax.spines.values(): sp.set_edgecolor('#333')
    ax.legend(facecolor='#1a1f2e', edgecolor='#555', labelcolor='white', fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='#0d1117')
    plt.close()
    print(f'  [Plot] ROC curves -> {save_path}')


def plot_confusion_matrix(y_true, y_pred, label_names, save_path):
    cm = confusion_matrix(y_true, y_pred)
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

    fig, ax = plt.subplots(figsize=(8,7))
    fig.patch.set_facecolor('#0d1117')
    ax.set_facecolor('#0d1117')
    im = ax.imshow(cm_pct, cmap='Blues')
    cbar = plt.colorbar(im, ax=ax)
    cbar.ax.tick_params(colors='white')

    ax.set_xticks(range(len(label_names)))
    ax.set_yticks(range(len(label_names)))
    ax.set_xticklabels(label_names, rotation=45, ha='right', color='white', fontsize=10)
    ax.set_yticklabels(label_names, color='white', fontsize=10)

    for i in range(len(label_names)):
        for j in range(len(label_names)):
            v = cm_pct[i,j]
            ax.text(j,i,f'{v:.1f}%', ha='center',va='center',
                    color='white' if v < 50 else '#111', fontsize=9)

    ax.set_xlabel('Predicted', color='white', fontsize=12)
    ax.set_ylabel('True',      color='white', fontsize=12)
    ax.set_title('V14.0 - Confusion Matrix (%)', color='white', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='#0d1117')
    plt.close()
    print(f'  [Plot] Confusion matrix -> {save_path}')


# ═══════════════════════════════════════════════════════════════
# SAVE MODEL
# ═══════════════════════════════════════════════════════════════
def save_artifacts(model, splits, thresholds, history, centroids, save_dir, hybrid_meta=None):
    os.makedirs(save_dir, exist_ok=True)

    pth_path = os.path.join(save_dir, 'ids_v14_model.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'n_features':       splits['n_features'],
        'n_classes':        splits['n_classes'],
        'hidden':           model.backbone.hidden,      # Luu lai de load dung architecture
        'ae_hidden':        model.ae.enc[4].in_features, # enc[4]=Linear(mid, mid//2) -> in_features=mid=ae_hidden
        'label_classes':    list(splits['label_encoder'].classes_),
        'known_cats':       splits['known_cats'],
        'zd_cats':          splits['zd_cats'],
        'feat_cols':        splits['feat_cols'],
        'categorical_maps': splits.get('categorical_maps', {}),
        'thresholds':       thresholds,
        'hybrid_meta':      hybrid_meta,
        'history':          history,
        'version':          'v14.0',
    }, pth_path)
    print(f'  Model weights -> {pth_path}')

    pkl_path = os.path.join(save_dir, 'ids_v14_pipeline.pkl')
    pipeline = {
        'scaler':        splits['scaler'],
        'label_encoder': splits['label_encoder'],
        'feat_cols':     splits['feat_cols'],
        'feature_names': splits['feat_cols'],
        'known_cats':    splits['known_cats'],
        'zd_cats':       splits['zd_cats'],
        'thresholds':    thresholds,
        'hybrid_meta':   hybrid_meta,
        'categorical_maps': splits.get('categorical_maps', {}),
        'centroids_np':  centroids.cpu().numpy(),
        'n_features':    splits['n_features'],
        'n_classes':     splits['n_classes'],
        'version':       'v14.0',
    }
    with open(pkl_path,'wb') as f:
        pickle.dump(pipeline, f)
    print(f'  Pipeline pkl  -> {pkl_path}')
    return pth_path, pkl_path


# ═══════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════
def run_full(args):
    print('\n'+'='*70)
    print('IDS v14.0 - Hybrid GradBP+AE  |  UNSW-NB15')
    print('='*70)

    seed_everything(args.seed)

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.plot_dir, exist_ok=True)

    print('\n[1/8] Loading data...')
    df = load_unsw_csvs(args.data_dir)
    df = clean_df(df)

    print('\n[2/8] Preparing splits...')
    splits = prepare_splits(df, seed=args.seed,
                            zd_augment=getattr(args,'zd_augment_factor',1))

    le = splits['label_encoder']
    label_names = list(le.classes_)
    dos_idx = None
    recon_idx = None
    if 'DoS' in label_names:
        dos_idx = int(le.transform(['DoS'])[0])
    if 'Reconnaissance' in label_names:
        recon_idx = int(le.transform(['Reconnaissance'])[0])

    print('\n[3/8] Creating loaders...')
    loaders = make_loaders(splits, batch_size=args.batch_size,
                           num_workers=args.num_workers,
                           dos_class_idx=dos_idx, dos_over=5.0,
                           seed=args.seed)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'\n[4/8] Building model (device={device})...')
    model = IDSModel(n_features=splits['n_features'],
                     n_classes=splits['n_classes'],
                     hidden=args.hidden, ae_hidden=args.ae_hidden)
    model = model.to(device)

    criterion = IDSLoss(
        n_classes=splits['n_classes'], lambda_con=args.lambda_con,
        focal_gamma=args.focal_gamma,  dos_class_idx=dos_idx,
        dos_weight=args.dos_weight,    recon_class_idx=recon_idx,
        recon_dos_penalty=getattr(args, 'recon_dos_penalty', 2.0),
        n_features=splits['n_features'],
        device=device,
    )

    print('\n[5/8] Training...')
    model, history = train(model, loaders, args, criterion, device, label_names=label_names)

    print('\n[6/8] Building centroids + calibrating thresholds...')
    hybrid_meta = fit_hybrid_meta_learner(
        model, splits['X_val'], splits['X_zd'], device, seed=args.seed,
    )
    centroids  = build_centroids(model, splits['X_train'], splits['y_train'],
                                  n_clusters=args.n_clusters, device=device,
                                  seed=args.seed)
    thresholds = calibrate(model, splits['X_val'], splits['y_val'],
                           args.target_fpr, device, centroids,
                           hybrid_meta=hybrid_meta)
    thresholds['hybrid_meta'] = hybrid_meta

    normal_idx  = label_names.index('Normal') if 'Normal' in label_names else 0
    model.eval()
    re_val = []
    with torch.no_grad():
        for i in range(0,len(splits['X_val']),512):
            x = torch.FloatTensor(splits['X_val'][i:i+512]).to(device)
            re_val.append(model.ae.recon_error(x).cpu().numpy())
    re_val = np.concatenate(re_val)
    re_thr = float(np.quantile(re_val, 1-args.target_fpr))
    threshold_trace = None
    final_ae_threshold = float(thresholds.get('ae_re', re_thr))
    if getattr(args, 'adaptive_threshold', False):
        threshold_trace = compute_adaptive_threshold_trace(
            model, splits['X_test'], splits['y_test'], normal_idx,
            re_val, args.target_fpr, device,
        )
        final_ae_threshold = float(threshold_trace['final_threshold'])
        thresholds['ae_re'] = final_ae_threshold
        re_thr = final_ae_threshold
        print(f'\n  Adaptive AE threshold final={final_ae_threshold:.6f}')
    p_thr  = 0.5

    print('\n[7/8] Evaluating...')
    clf_res     = evaluate_classifier(model, splits['X_test'], splits['y_test'],
                                       label_names, device)
    clf_res['y_test'] = splits['y_test']

    zd_res = evaluate_zero_day(
        model, splits['X_test'], splits['y_test'],
        splits['X_zd'],          splits['y_zd'],
        thresholds, centroids, device, hybrid_meta=hybrid_meta,
    )

    print('\n[8/8] Saving & plotting...')
    pth_p, pkl_p = save_artifacts(
        model, splits, thresholds, history, centroids, args.save_dir,
        hybrid_meta=hybrid_meta,
    )

    plot_training_curve(history, os.path.join(args.plot_dir,'v14_training_curve.png'))
    plot_soc_decision_space(model, splits['X_test'], splits['y_test'],
        splits['X_zd'], p_thr, re_thr, device,
        os.path.join(args.plot_dir,'v14_decision_space.png'), label_names,
        seed=args.seed)

    per_cls_zd   = zd_res.get('_per_class',{})
    zd_cls_order = sorted(per_cls_zd.keys())
    plot_per_class_proper(label_names, splits['y_test'], clf_res['preds'],
        per_cls_zd, zd_cls_order,
        os.path.join(args.plot_dir,'v14_per_class_detection.png'), normal_idx=normal_idx)
    plot_roc_curves(zd_res, os.path.join(args.plot_dir,'v14_roc_curves.png'))
    plot_confusion_matrix(splits['y_test'], clf_res['preds'], label_names,
                          os.path.join(args.plot_dir,'v14_confusion_matrix.png'))
    if threshold_trace is not None:
        plot_threshold_drift(
            threshold_trace,
            os.path.join(args.plot_dir, 'v14_threshold_drift.png'),
        )

    best_zd_auc = max((v['auc'] for k,v in zd_res.items()
                       if not k.startswith('_') and 'auc' in v), default=0.)

    results_dir = os.path.abspath(os.path.join(args.save_dir, '..', 'results'))
    os.makedirs(results_dir, exist_ok=True)
    summary = {
        'version': 'v14.0',
        'known_auc': float(clf_res['auc']),
        'best_zd_auc': float(best_zd_auc),
        'best_zd_method': zd_res.get('_best_method'),
        'n_features': int(splits['n_features']),
        'n_classes': int(splits['n_classes']),
        'n_epochs': len(history),
        'thresholds': thresholds,
        'hybrid_meta': hybrid_meta,
        'final_ae_threshold': final_ae_threshold,
        'known_cats': splits['known_cats'],
        'zd_cats': splits['zd_cats'],
    }
    with open(os.path.join(results_dir, 'ids_v14_results.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, default=str)

    print(f'\n{"="*70}')
    print('FINAL SUMMARY - IDS v14.0')
    print(f'{"="*70}')
    print(f'  Known AUC   : {clf_res["auc"]:.4f}')
    print(f'  Best ZD AUC : {best_zd_auc:.4f}')
    print(f'  Model .pth  : {pth_p}')
    print(f'  Pipeline    : {pkl_p}')
    print(f'{"="*70}')

    return model, zd_res, history


# ═══════════════════════════════════════════════════════════════
# DEMO MODE
# ═══════════════════════════════════════════════════════════════
def run_demo(args):
    print('\n'+'='*60)
    print('DEMO MODE - Synthetic UNSW-NB15-like data')
    print('='*60)
    seed_everything(getattr(args, 'seed', 42))
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.plot_dir, exist_ok=True)

    n_feat=55; n_cls=5; n_zd=5
    N=60000
    X = np.random.randn(N, n_feat).astype(np.float32)
    y = np.random.randint(0, n_cls, N)
    for c in range(n_cls): X[y==c] += c*1.5

    dos_idx_demo = 1
    X[y==dos_idx_demo] = np.random.randn((y==dos_idx_demo).sum(), n_feat)*0.8

    N_zd = 8000
    X_zd = (np.random.randn(N_zd, n_feat)*1.5+3.).astype(np.float32)
    y_zd = np.array([f'ZD_{i%n_zd}' for i in range(N_zd)])

    X_tv,X_te,y_tv,y_te = train_test_split(X,y,test_size=0.2,stratify=y)
    X_tr,X_va,y_tr,y_va = train_test_split(X_tv,y_tv,test_size=0.125,stratify=y_tv)

    sc = RobustScaler().fit(X_tr)
    X_tr = sc.transform(X_tr); X_va = sc.transform(X_va)
    X_te = sc.transform(X_te); X_zd = sc.transform(X_zd)

    le = LabelEncoder()
    le.classes_ = np.array([f'Class_{i}' for i in range(n_cls)])

    splits = dict(
        X_train=X_tr, y_train=y_tr,
        X_val=X_va,   y_val=y_va,
        X_test=X_te,  y_test=y_te,
        X_zd=X_zd,    y_zd=y_zd,
        n_features=n_feat, n_classes=n_cls,
        label_encoder=le, scaler=sc,
        feat_cols=[f'f{i}' for i in range(n_feat)],
        feature_names=[f'f{i}' for i in range(n_feat)],
        known_cats=[f'Class_{i}' for i in range(n_cls)],
        zd_cats=[f'ZD_{i}' for i in range(n_zd)],
    )

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model  = IDSModel(n_features=n_feat, n_classes=n_cls, hidden=128, ae_hidden=64)
    model  = model.to(device)

    args.epochs   = min(getattr(args,'epochs',10), 10)
    args.patience = min(getattr(args,'patience',5), 5)
    args.lr       = getattr(args, 'lr', 3e-4)
    args.hidden   = getattr(args, 'hidden', 128)

    criterion = IDSLoss(n_classes=n_cls, lambda_con=0.3, focal_gamma=2.0,
                        dos_class_idx=dos_idx_demo, dos_weight=getattr(args, 'dos_weight', 3.0),
                        device=device)
    loaders = make_loaders(splits, batch_size=256, num_workers=0,
                           dos_class_idx=dos_idx_demo,
                           seed=getattr(args, 'seed', 42))
    label_names = [f'Class_{i}' for i in range(n_cls)]
    model, history = train(model, loaders, args, criterion, device, label_names=label_names)

    hybrid_meta = fit_hybrid_meta_learner(
        model, X_va, X_zd, device, seed=getattr(args, 'seed', 42),
    )
    centroids  = build_centroids(model, X_tr, y_tr, 10, device,
                                 seed=getattr(args, 'seed', 42))
    thresholds = calibrate(
        model, X_va, y_va, args.target_fpr, device, centroids,
        hybrid_meta=hybrid_meta,
    )
    thresholds['hybrid_meta'] = hybrid_meta
    re_val_demo = _batch_scores(model, X_va, device)['ae_re']
    re_thr_demo = float(np.quantile(re_val_demo, 1 - args.target_fpr))
    threshold_trace = None
    if getattr(args, 'adaptive_threshold', False):
        threshold_trace = compute_adaptive_threshold_trace(
            model, X_te, y_te, 0, re_val_demo, args.target_fpr, device,
        )
        thresholds['ae_re'] = float(threshold_trace['final_threshold'])
        re_thr_demo = float(threshold_trace['final_threshold'])
        print(f'\n  Adaptive AE threshold final={re_thr_demo:.6f}')

    clf_res     = evaluate_classifier(model, X_te, y_te, label_names, device)
    clf_res['y_test'] = y_te
    zd_res      = evaluate_zero_day(model, X_te, y_te, X_zd, y_zd,
                                     thresholds, centroids, device,
                                     hybrid_meta=hybrid_meta)

    pth_p, pkl_p = save_artifacts(
        model, splits, thresholds, history, centroids, args.save_dir,
        hybrid_meta=hybrid_meta,
    )

    plot_training_curve(history, os.path.join(args.plot_dir,'v14_training_curve.png'))
    plot_soc_decision_space(model,X_te,y_te,X_zd,0.5,re_thr_demo,device,
                             os.path.join(args.plot_dir,'v14_decision_space.png'),label_names,
                             seed=getattr(args, 'seed', 42))
    per_cls_zd   = zd_res.get('_per_class',{})
    zd_cls_order = sorted(per_cls_zd.keys())
    plot_per_class_proper(label_names, y_te, clf_res['preds'], per_cls_zd, zd_cls_order,
                           os.path.join(args.plot_dir,'v14_per_class_detection.png'), normal_idx=0)
    plot_roc_curves(zd_res, os.path.join(args.plot_dir,'v14_roc_curves.png'))
    plot_confusion_matrix(y_te, clf_res['preds'], label_names,
                          os.path.join(args.plot_dir,'v14_confusion_matrix.png'))
    if threshold_trace is not None:
        plot_threshold_drift(
            threshold_trace,
            os.path.join(args.plot_dir, 'v14_threshold_drift.png'),
        )

    print(f'\nDemo done!  Plots -> {args.plot_dir}')
    print(f'   .pth -> {pth_p}')
    print(f'   .pkl -> {pkl_p}')
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
