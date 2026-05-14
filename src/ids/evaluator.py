"""OOD scoring, centroid construction, calibration and zero-day evaluation helpers."""

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import MiniBatchKMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve, classification_report

from .threshold import AdaptiveThreshold

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


def predict_with_uncertainty(model, x, n_samples=30):
    """
    Estimate predictive uncertainty with Monte Carlo Dropout.

    This intentionally enables training mode for the sampled forward passes so
    Dropout layers stay stochastic, then restores the model's previous mode.
    """
    if n_samples <= 0:
        raise ValueError('n_samples must be positive')

    was_training = bool(model.training)
    model.train()
    samples = []
    try:
        with torch.no_grad():
            for _ in range(int(n_samples)):
                outputs = model(x)
                logits = outputs[0] if isinstance(outputs, tuple) else outputs
                samples.append(torch.softmax(logits, dim=-1))
    finally:
        if not was_training:
            model.eval()

    stacked = torch.stack(samples, dim=0)
    mean_probs = stacked.mean(dim=0).squeeze(0)
    std_probs = stacked.std(dim=0, unbiased=False).squeeze(0)
    entropy = -(mean_probs * torch.log(mean_probs.clamp_min(1e-12))).sum()
    return mean_probs.detach().cpu(), std_probs.detach().cpu(), float(entropy.detach().cpu().item())


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
