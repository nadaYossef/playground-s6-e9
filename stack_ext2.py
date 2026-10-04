import numpy as np,pandas as pd,glob
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score as A
from sklearn.model_selection import StratifiedKFold
tr=pd.read_csv('train.csv',usecols=['id','Will_Buy_EV']); te=pd.read_csv('test.csv',usecols=['id'])
TI,EI=tr.id.values,te.id.values; y=(tr.Will_Buy_EV=='Yes').astype(int).values
rk=lambda v: rankdata(v)/len(v)
def al(path,col,ids):
    d=pd.read_parquet(path) if path.endswith('parquet') else pd.read_csv(path)
    s=d.set_index('id')[col].reindex(ids); assert s.notna().all(),(path,col,s.isna().sum()); return s.values
E='ext/'
M={}
M['heuljax_lr']=(al(E+'kps6e09-generator-aware-ridge-logistic-regression/oof/GENERATOR_AWARE_LOGREG_SAMPLE_OOF.parquet','oof_pred',TI),al(E+'kps6e09-generator-aware-ridge-logistic-regression/test_preds/GENERATOR_AWARE_LOGREG_SAMPLE_TEST.parquet','test_pred',EI))
M['heuljax_xgb']=(al(E+'kps6e09-xgb-sample/oof/XGB_SAMPLE_OOF.parquet','oof_pred',TI),al(E+'kps6e09-xgb-sample/test_preds/XGB_SAMPLE_TEST.parquet','test_pred',EI))
M['blamerx']=(al(E+'s6e9-xgboost-window-encodings-0-946-cv/oof.csv','pred',TI),al(E+'s6e9-xgboost-window-encodings-0-946-cv/submission.csv','Will_Buy_EV',EI))
V=E+'s6e9-six-feature-views-oof-library/'
for c in [c for c in pd.read_csv(V+'oof_six_views.csv',nrows=2).columns if c not in ('id','fold','Will_Buy_EV','y','target')]:
    M['mg_'+c[:30]]=(al(V+'oof_six_views.csv',c,TI),al(V+'test_six_views.csv',c,EI))
M['realmlp']=(al(V+'oof_realmlp_g.csv','G_realmlp_3seed',TI),al(V+'test_realmlp_g.csv','G_realmlp_3seed',EI))
G=E+'s6e9-lr-margin-gbdt-oof-stack-lb-0-94675/'
for c in ['glm','residual_lgbm','residual_xgb']: M['gp_'+c]=(al(G+'oof_mine.csv',c,TI),al(G+'test_mine.csv',c,EI))
Q=['Q_lgb_s0_f10_k1_b255','Q_lgb_s1_f10_k1_b255','Q_xgb_s0_f10_k1_b255','Q_xgb_s1_f10_k1_b255']
M['mine_Q']=(np.mean([rk(np.load(f'preds/{n}_oof.npy')) for n in Q],0),np.mean([rk(np.load(f'preds/{n}_test.npy')) for n in Q],0))
names=list(M); R={n:rk(M[n][0]) for n in names}; T={n:rk(M[n][1]) for n in names}
for n in names: print(f'{n:34s} OOF {A(y,R[n]):.5f}  corr w/ mine {np.corrcoef(R[n],R["mine_Q"])[0,1]:.4f}')
def hill(Rs,yy,step=0.02,iters=300):
    w={n:0.0 for n in names}; b=max(names,key=lambda n:A(yy,Rs[n])); w[b]=1; v=Rs[b].copy(); cur=A(yy,v)
    for _ in range(iters):
        a,n=max((A(yy,v+step*Rs[n]),n) for n in names)
        if a<=cur+1e-7: break
        w[n]+=step; v=v+step*Rs[n]; cur=a
    s=sum(w.values()); return {n:w[n]/s for n in names}
ws=[]; gains=[]
for i,(a,b) in enumerate(StratifiedKFold(5,shuffle=True,random_state=0).split(np.zeros(len(y)),y)):
    wi=hill({n:R[n][a] for n in names},y[a]); ws.append(wi)
    held=A(y[b],sum(wi[n]*R[n][b] for n in names)); mine=A(y[b],R['mine_Q'][b]); best=max(A(y[b],R[n][b]) for n in names)
    gains.append(held-mine); print(f'nested fold {i}: mine {mine:.5f} best single {best:.5f} stack {held:.5f} gain vs mine {held-mine:+.5f}',flush=True)
W={n:np.mean([w[n] for w in ws]) for n in names}
print('weights',{n:round(v,3) for n,v in sorted(W.items(),key=lambda x:-x[1]) if v>0.004})
so=sum(W[n]*R[n] for n in names); st=sum(W[n]*T[n] for n in names)
print('STACK OOF',round(A(y,so),5),'mine',round(A(y,R['mine_Q']),5),'gain positive in',sum(g>0 for g in gains),'/5 folds')
pd.DataFrame({'id':EI,'Will_Buy_EV':st}).to_csv('submission7.csv',index=False); print('saved submission7.csv')
