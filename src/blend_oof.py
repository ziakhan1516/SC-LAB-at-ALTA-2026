"""Fit the ensemble on OUT-OF-FOLD predictions and apply it to the blind test set.

The dev-phase winner fitted weights+thresholds against valid.csv, whose labels are public.
Here the target labels are blind, so everything is fitted on the 6288 OOF rows instead and
an inner CV reports what each variant is actually worth out of sample.
"""
import json, sys, glob, os
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold

W = "."
tr = pd.read_csv('data/train.csv')
va = pd.read_csv('data/valid.csv')
pool = pd.concat([tr, va], ignore_index=True)          # exact order train_v8.py used
tgt = pd.read_csv('data/test.csv')
assert len(pool) == 6288 and len(tgt) == 1573

Yp = {'sentiment': pool.sentiment.values, 'sarcasm': pool.sarcasm.values}
vp, sp = pool.variety.values, pool.source.values
vt, st = tgt.variety.values, tgt.source.values

def rk(x):
    o = np.empty_like(x, dtype=float)
    for c in range(x.shape[1]):
        r = x[:, c].argsort().argsort().astype(float)
        o[:, c] = r / max(len(r) - 1, 1)
    return o

MODELS = {}
for f in sorted(glob.glob('artifacts/*_oof.npy')):
    stem = f[:-8]; name = os.path.basename(stem)
    t = stem + '_target.npy'
    if not os.path.exists(t) or 'quick' in name:
        continue
    o, tt = np.load(f), np.load(t)
    if o.shape[0] != len(pool) or tt.shape[0] != len(tgt):
        print(f'  skip {name}: shapes {o.shape} {tt.shape}'); continue
    MODELS[name] = (o, tt)
print(f'{len(MODELS)} models: ' + ', '.join(sorted(MODELS)))

# feature space: rank / raw prob / cross-task rank / inverted rank, per model
def featurize(idx_col):
    """returns names, F_oof (n_feat x 6288), F_tgt (n_feat x 1573) for task column idx_col"""
    names, fo, ft = [], [], []
    other = 1 - idx_col
    for k, (o, t) in sorted(MODELS.items()):
        Ro, Rt = rk(o), rk(t)
        for kind, a, b in (('rank',   Ro[:, idx_col], Rt[:, idx_col]),
                           ('prob',   o[:, idx_col],  t[:, idx_col]),
                           ('xrank',  Ro[:, other],   Rt[:, other]),
                           ('inv',    1 - Ro[:, idx_col], 1 - Rt[:, idx_col])):
            names.append(f'{k}|{kind}'); fo.append(a); ft.append(b)
    return names, np.stack(fo), np.stack(ft)

GRID = np.round(np.arange(0.01, 1.0, 0.005), 5)

def macro_f1(y, p):
    out = []
    for c in (0, 1):
        tp = ((y == c) & (p == c)).sum(); fp = ((y != c) & (p == c)).sum(); fn = ((y == c) & (p != c)).sum()
        out.append(0.0 if 2*tp + fp + fn == 0 else 2*tp / (2*tp + fp + fn))
    return float(np.mean(out))

def cnt(y, p):
    return (int(((y == 1) & (p == 1)).sum()), int(((y == 0) & (p == 1)).sum()),
            int(((y == 1) & (p == 0)).sum()), int(((y == 0) & (p == 0)).sum()))

def sweep(score, y, fx=(0, 0, 0, 0)):
    """macro-F1 at EVERY threshold in GRID at once, given the other source's fixed counts.
    Sorting once + a cumulative sum replaces an O(|GRID| * n) python loop with O(n log n)."""
    o = np.argsort(score, kind='stable')
    ps, ys = score[o], y[o].astype(np.int64)
    P = int(ys.sum()); n = len(ys)
    cp = np.concatenate([[0], np.cumsum(ys)])
    k = np.searchsorted(ps, GRID, side='left')
    TP = P - cp[k] + fx[0]
    FP = (n - k) - (P - cp[k]) + fx[1]
    FN = cp[k] + fx[2]
    TN = (k - cp[k]) + fx[3]
    z = np.zeros(len(GRID))
    u = np.divide(2*TP, 2*TP + FP + FN, out=z.copy(), where=(2*TP + FP + FN) > 0)
    v = np.divide(2*TN, 2*TN + FN + FP, out=z.copy(), where=(2*TN + FN + FP) > 0)
    return (u + v) / 2

def sweep_thr(score, y, fx=(0, 0, 0, 0)):
    c = sweep(score, y, fx); i = int(np.argmax(c))
    return float(c[i]), float(GRID[i])

AGG = {'mean':   lambda X, w: (w @ X) / w.sum(),
       'median': lambda X, w: np.median(X[w > 0], axis=0),
       'max':    lambda X, w: X[w > 0].max(axis=0),
       'min':    lambda X, w: X[w > 0].min(axis=0)}

def fit_cell(F, y, src, var_mask, restarts, seed, allow_agg):
    """coordinate ascent over integer weights x aggregation x threshold, per source"""
    SRC = {s: var_mask & (src == s) for s in ('google', 'reddit')}
    SRC = {s: m for s, m in SRC.items() if m.sum() > 0}
    Y = {s: y[m] for s, m in SRC.items()}
    M = {s: F[:, m] for s, m in SRC.items()}
    rng = np.random.default_rng(seed)
    nF = F.shape[0]
    best = (-1, None)
    for _ in range(restarts):
        w = {s: np.zeros(nF) for s in SRC}
        agg = {s: 'mean' for s in SRC}
        for s in SRC: w[s][rng.integers(nF)] = 1
        thr, pred, cur = {}, {}, -1
        for s in SRC:
            pt = AGG[agg[s]](M[s], w[s])
            _, thr[s] = sweep_thr(pt, Y[s])
            pred[s] = (pt >= thr[s]).astype(int)
        for _it in range(30):
            moved = False
            for s in SRC:
                oth = [x for x in SRC if x != s]
                fx = tuple(sum(v) for v in zip(*[cnt(Y[o], pred[o]) for o in oth])) if oth else (0, 0, 0, 0)
                bb, bw, bt, ba = cur, None, None, None
                for ki in range(nF):
                    for d in (1, -1):
                        if d < 0 and w[s][ki] <= 0: continue
                        w2 = w[s].copy(); w2[ki] += d
                        if w2.sum() <= 0: continue
                        for an in (AGG if allow_agg else {'mean': AGG['mean']}):
                            pt = AGG[an](M[s], w2)
                            f, t = sweep_thr(pt, Y[s], fx)
                            if f > bb + 1e-12: bb, bw, bt, ba = f, w2, t, an
                if bw is not None:
                    w[s], thr[s], agg[s], cur = bw, bt, ba, bb
                    pred[s] = (AGG[agg[s]](M[s], w[s]) >= thr[s]).astype(int)
                    moved = True
            if not moved: break
        if cur > best[0]:
            best = (cur, {'agg': dict(agg), 'thr': {k: float(v) for k, v in thr.items()},
                          'w': {s: {int(i): int(w[s][i]) for i in np.nonzero(w[s])[0]} for s in SRC}})
    return best

def apply_cell(spec, names, Ft, src, var_mask):
    p = np.zeros(var_mask.sum(), int)
    sub = src[var_mask]
    for s, wd in spec['w'].items():
        m = sub == s
        if m.sum() == 0: continue
        idx = [int(i) for i in wd]; wv = np.array([wd[i] for i in wd], float)
        X = Ft[np.ix_(idx, np.where(var_mask)[0])][:, m]
        p[m] = (AGG[spec['agg'][s]](X, wv) >= spec['thr'][s]).astype(int)
    return p

# --------------------------------------------------------------------------- #
VARIANTS = {
  'A_rankavg':  dict(kinds=('rank',),                       restarts=0,  agg=False),
  'B_subset':   dict(kinds=('rank',),                       restarts=60, agg=False),
  'C_expanded': dict(kinds=('rank','prob','xrank','inv'),   restarts=60, agg=True),
}
TASKS = [('sentiment', 0), ('sarcasm', 1)]
VARS = ['en-AU', 'en-UK']

def select_feats(names, F, kinds):
    keep = [i for i, n in enumerate(names) if n.split('|')[1] in kinds]
    return [names[i] for i in keep], F[keep]

def rank_average_cell(F, y, src, var_mask):
    """variant A: unweighted mean of every feature, threshold tuned per source on OOF"""
    SRC = {s: var_mask & (src == s) for s in ('google', 'reddit')}
    SRC = {s: m for s, m in SRC.items() if m.sum() > 0}
    w = np.ones(F.shape[0])
    thr, pred = {}, {}
    for s, m in SRC.items():
        pt = AGG['mean'](F[:, m], w)
        _, thr[s] = sweep_thr(pt, y[m])
        pred[s] = (pt >= thr[s]).astype(int)
    return {'agg': {s: 'mean' for s in SRC}, 'thr': thr,
            'w': {s: {int(i): 1 for i in range(F.shape[0])} for s in SRC}}

def fit_all(rows, cfg, seed=0):
    """fit every (task,variety) cell on the OOF rows in `rows`; return spec dict"""
    spec = {}
    for task, col in TASKS:
        names, Fo, _ = FEATS[task]
        names, Fo = select_feats(names, Fo, cfg['kinds'])
        for var in VARS:
            vm = np.zeros(len(pool), bool); vm[rows] = True
            vm &= (vp == var)
            y = Yp[task]
            if cfg['restarts'] == 0:
                s = rank_average_cell(Fo, y, sp, vm)
            else:
                _, s = fit_cell(Fo, y, sp, vm, cfg['restarts'], seed, cfg['agg'])
            spec[f'{task}-{var}'] = s
    return spec

def score_spec(spec, cfg, rows):
    """official score of `spec` on held-out OOF rows"""
    per = {}
    for task, col in TASKS:
        names, Fo, _ = FEATS[task]
        names, Fo = select_feats(names, Fo, cfg['kinds'])
        for var in VARS:
            vm = np.zeros(len(pool), bool); vm[rows] = True
            vm &= (vp == var)
            p = apply_cell(spec[f'{task}-{var}'], names, Fo, sp, vm)
            per[f'{task}-{var}'] = macro_f1(Yp[task][vm], p)
    fin = (min(per['sentiment-en-AU'], per['sentiment-en-UK'])
         + min(per['sarcasm-en-AU'],   per['sarcasm-en-UK'])) / 2
    return per, fin

FEATS = {}
for task, col in TASKS:
    FEATS[task] = featurize(col)

if __name__ == '__main__':
    strat = (pool.variety + '|' + pool.source + '|' + pool.sentiment.astype(str)
             + '|' + pool.sarcasm.astype(str)).values
    u, c = np.unique(strat, return_counts=True); rare = set(u[c < 5])
    strat = np.array(['|'.join(s.split('|')[:2]) if s in rare else s for s in strat])
    skf = StratifiedKFold(5, shuffle=True, random_state=2026)
    report = {}
    for vn, cfg in VARIANTS.items():
        # honest nested CV: choose the blend on 4/5 of the OOF rows, score on the held-out 1/5
        cv = []
        for k, (fit_i, ev_i) in enumerate(skf.split(np.zeros(len(pool)), strat)):
            sp_ = fit_all(fit_i, cfg, seed=k)
            _, f = score_spec(sp_, cfg, ev_i)
            cv.append(f)
        final_spec = fit_all(np.arange(len(pool)), cfg, seed=0)
        insample, fin_in = score_spec(final_spec, cfg, np.arange(len(pool)))
        report[vn] = {'nested_cv_mean': float(np.mean(cv)), 'nested_cv_sd': float(np.std(cv)),
                      'in_sample': fin_in, 'per_cell_in_sample': insample}
        print(f'{vn:<12} nested-CV {np.mean(cv):.4f} +/- {np.std(cv):.4f}   (in-sample {fin_in:.4f})')
        # predictions on the blind test set
        pred = {}
        for task, col in TASKS:
            names, _, Ft = FEATS[task]
            names, Ft = select_feats(names, Ft, cfg['kinds'])
            p = np.zeros(len(tgt), int)
            for var in VARS:
                vm = (vt == var)
                p[vm] = apply_cell(final_spec[f'{task}-{var}'], names, Ft, st, vm)
            pred[task] = p
        ans = tgt.copy(); ans['sentiment'] = pred['sentiment']; ans['sarcasm'] = pred['sarcasm']
        ans[['source','variety','text','sentiment','sarcasm']].to_csv(f'submissions/answer_{vn}.csv', index=False)
        json.dump({k: v for k, v in final_spec.items()}, open(f'submissions/spec_{vn}.json','w'), indent=2, default=str)
    json.dump(report, open('submissions/blend_report.json','w'), indent=2)
    print('\nwrote answer_*.csv + blend_report.json')
