"""Bagged greedy ensemble selection (Caruana 2004/2006), per (task,variety,source).

B_subset picked ONE subset by coordinate ascent on the full OOF -> high selection variance
(held-out 0.8621, leaderboard 0.8526). Bagging averages the selection over bootstrap
resamples, which is the standard variance fix and what v3 used.
"""
import json, glob, os, sys
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold

W='.'
tr=pd.read_csv('data/train.csv'); va=pd.read_csv('data/valid.csv')
pool=pd.concat([tr,va],ignore_index=True); tgt=pd.read_csv('data/test.csv')
vp,sp=pool.variety.values,pool.source.values; vt,st=tgt.variety.values,tgt.source.values
GRID=np.round(np.arange(0.01,1.0,0.0025),6)

def rk(x):
    o=np.empty_like(x,dtype=float)
    for c in range(x.shape[1]):
        r=x[:,c].argsort().argsort().astype(float); o[:,c]=r/max(len(r)-1,1)
    return o
NAMES,RO,RT=[],[],[]
for f in sorted(glob.glob('artifacts/*_oof.npy')):
    n=os.path.basename(f)[:-8]
    t=f[:-8]+'_target.npy'
    if 'quick' in n or not os.path.exists(t): continue
    o,tt=np.load(f),np.load(t)
    if o.shape[0]!=len(pool) or tt.shape[0]!=len(tgt): continue
    NAMES.append(n); RO.append(rk(o)); RT.append(rk(tt))
M=len(NAMES); print(f'{M} models',flush=True)

def mf1(y,p):
    s=0.0
    for c in (0,1):
        tp=((y==c)&(p==c)).sum(); fp=((y!=c)&(p==c)).sum(); fn=((y==c)&(p!=c)).sum()
        s+= 0.0 if 2*tp+fp+fn==0 else 2*tp/(2*tp+fp+fn)
    return s/2

def sweep(score,y,fx=(0,0,0,0)):
    o=np.argsort(score,kind='stable'); ps,ys=score[o],y[o].astype(np.int64)
    P=int(ys.sum()); n=len(ys); cp=np.concatenate([[0],np.cumsum(ys)])
    k=np.searchsorted(ps,GRID,side='left')
    TP=P-cp[k]+fx[0]; FP=(n-k)-(P-cp[k])+fx[1]; FN=cp[k]+fx[2]; TN=(k-cp[k])+fx[3]
    z=np.zeros(len(GRID))
    a=np.divide(2*TP,2*TP+FP+FN,out=z.copy(),where=(2*TP+FP+FN)>0)
    b=np.divide(2*TN,2*TN+FN+FP,out=z.copy(),where=(2*TN+FN+FP)>0)
    return (a+b)/2

def bagged_weights(X,y,bags,steps,rng):
    """X: (M,n) ranks for this cell+source. Returns averaged selection counts."""
    n=X.shape[1]; W_=np.zeros(M)
    for _ in range(bags):
        idx=rng.integers(0,n,n)                       # bootstrap rows
        Xb,yb=X[:,idx],y[idx]
        cnt=np.zeros(M); cur=None; best_f=-1
        for _s in range(steps):
            bi,bf=-1,best_f
            for m in range(M):                        # forward selection WITH replacement
                cand=(cur*cnt.sum()+Xb[m])/(cnt.sum()+1) if cur is not None else Xb[m]
                f=sweep(cand,yb).max()
                if f>bf+1e-12: bf,bi=f,m
            if bi<0: break
            cnt[bi]+=1
            cur=(cur*(cnt.sum()-1)+Xb[bi])/cnt.sum() if cur is not None else Xb[bi]
            best_f=bf
        if cnt.sum()>0: W_+=cnt/cnt.sum()
    return W_/max(W_.sum(),1e-12)

def fit(rows,bags,steps,seed):
    rng=np.random.default_rng(seed); spec={}
    for task,col in (('sentiment',0),('sarcasm',1)):
        y_all=pool[task].values
        for var in ('en-AU','en-UK'):
            cell={}
            base=np.zeros(len(pool),bool); base[rows]=True; base&=(vp==var)
            for src in ('google','reddit'):
                m=base&(sp==src)
                if m.sum()==0: continue
                X=np.stack([R[m][:,col] for R in RO]); y=y_all[m]
                w=bagged_weights(X,y,bags,steps,rng)
                s=w@X
                c=sweep(s,y); thr=float(GRID[int(np.argmax(c))])
                cell[src]={'w':w.tolist(),'thr':thr}
            spec[f'{task}-{var}']=cell
    return spec

def predict(spec,rows,which):
    """which='oof'|'tgt'"""
    R = RO if which=='oof' else RT
    V,S,n = (vp,sp,len(pool)) if which=='oof' else (vt,st,len(tgt))
    out={}
    for task,col in (('sentiment',0),('sarcasm',1)):
        p=np.zeros(n,int)
        for var in ('en-AU','en-UK'):
            for src,cf in spec[f'{task}-{var}'].items():
                m=(V==var)&(S==src)
                if rows is not None:
                    keep=np.zeros(n,bool); keep[rows]=True; m&=keep
                if m.sum()==0: continue
                X=np.stack([Rr[m][:,col] for Rr in R]); w=np.array(cf['w'])
                p[m]=((w@X)>=cf['thr']).astype(int)
        out[task]=p
    return out

def official(rows,pred):
    per={}
    for task in ('sentiment','sarcasm'):
        for var in ('en-AU','en-UK'):
            m=np.zeros(len(pool),bool); m[rows]=True; m&=(vp==var)
            per[f'{task}-{var}']=mf1(pool[task].values[m],pred[task][m])
    return per,(min(per['sentiment-en-AU'],per['sentiment-en-UK'])
               +min(per['sarcasm-en-AU'],per['sarcasm-en-UK']))/2

if __name__=='__main__':
    BAGS,STEPS=int(sys.argv[1]),int(sys.argv[2])
    strat=(pool.variety+'|'+pool.source+'|'+pool.sentiment.astype(str)+'|'+pool.sarcasm.astype(str)).values
    u,c=np.unique(strat,return_counts=True); rare=set(u[c<5])
    strat=np.array(['|'.join(s.split('|')[:2]) if s in rare else s for s in strat])
    cv=[]
    for k,(fi,ei) in enumerate(StratifiedKFold(5,shuffle=True,random_state=2026).split(np.zeros(len(pool)),strat)):
        sp_=fit(fi,BAGS,STEPS,seed=k)
        _,f=official(ei,predict(sp_,ei,'oof')); cv.append(f)
        print(f'  fold{k} {f:.4f}',flush=True)
    print(f'BAGGED nested-CV {np.mean(cv):.4f} +/- {np.std(cv):.4f}',flush=True)
    final=fit(np.arange(len(pool)),BAGS,STEPS,seed=0)
    pr=predict(final,None,'tgt')
    ans=tgt.copy(); ans['sentiment']=pr['sentiment']; ans['sarcasm']=pr['sarcasm']
    ans[['source','variety','text','sentiment','sarcasm']].to_csv('submissions/answer_D_bagged.csv',index=False)
    json.dump({'nested_cv':float(np.mean(cv)),'sd':float(np.std(cv)),'bags':BAGS,'steps':STEPS,
               'models':NAMES},open('submissions/bag_report.json','w'),indent=2)
    print('wrote answer_D_bagged.csv',flush=True)
