"""Entry point for IDS v14 UNSW-NB15 training, evaluation, artifact export and plotting."""

import os, json, pickle
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, LabelEncoder

from ids.config import CFG, get_config, seed_everything
from ids.dataset import load_unsw_csvs, clean_df, prepare_splits, make_loaders
from ids.models import IDSModel
from ids.losses import IDSLoss
from ids.trainer import train
from ids.evaluator import (
    _batch_scores,
    fit_hybrid_meta_learner,
    build_centroids,
    calibrate,
    compute_adaptive_threshold_trace,
    evaluate_classifier,
    evaluate_zero_day,
)
from ids.plots import (
    plot_training_curve,
    plot_soc_decision_space,
    plot_per_class_proper,
    plot_roc_curves,
    plot_confusion_matrix,
    plot_threshold_drift,
)

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


def main():
    cfg = get_config()
    if cfg.demo:
        return run_demo(cfg)
    return run_full(cfg)


if __name__ == '__main__':
    main()
