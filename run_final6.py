import sys,numpy as np,pandas as pd,warnings;warnings.filterwarnings('ignore')
from common import *; from fe2 import add_art
from sklearn.metrics import roc_auc_score as A
from sklearn.model_selection import StratifiedKFold
name=sys.argv[1]; seed=int(sys.argv[2]); ONLY=int(sys.argv[3]); NF=int(sys.argv[4]); XK=int(sys.argv[5]); MB=int(sys.argv[6]); tag=f'Q_{name}_s{seed}_f{NF}_k{XK}_b{MB}'
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression as LR2
X0,y,T0,ids=load(0); X,T=add_art(X0,T0); AL=pd.concat([X,T],ignore_index=True); n=len(X); nte=len(T)
RAW=pd.concat([X0,T0],ignore_index=True)
ok=(AL.Annual_Income_USD!=30000)
VALS=['Age','Environmental_Concern_Level','Daily_Commute_km','Subsidy_Available','Range_Anxiety_Level','Home_Charging_Possible','Charging_Stations_Near_Home','Number_of_Cars_Owned']
G=AL.loc[ok].groupby('Annual_Income_USD')
for c in VALS:
    m=G[c].transform('mean'); AL.loc[ok,'g_'+c]=m; AL.loc[ok,'d_'+c]=AL.loc[ok,c]-m
AL.loc[ok,'g_size']=G['Age'].transform('size')
for c in ['Annual_Income_USD','Daily_Commute_km','Age']:
    ctr=RAW[c].iloc[:n].value_counts(); cte=RAW[c].iloc[n:].value_counts()
    AL[c+'_ctr']=RAW[c].map(ctr).fillna(0).values; AL[c+'_cte']=RAW[c].map(cte).fillna(0).values
    AL[c+'_one_side']=((AL[c+'_ctr']==0)|(AL[c+'_cte']==0)).astype(int)
KEYS={'te_inc':('Annual_Income_USD',None,5),'te_com':('Daily_Commute_km',None,20),'te_age':('Age',None,20),
 'te_inc_b500':('Annual_Income_USD',500,20),'te_inc_b2000':('Annual_Income_USD',2000,20),'te_inc_b5000':('Annual_Income_USD',5000,20),
 'te_com_b1':('Daily_Commute_km',1,20),'te_com_b5':('Daily_Commute_km',5,20),
 'te_env':('Environmental_Concern_Level',None,20),'te_csh':('Charging_Stations_Near_Home',None,20),'te_csw':('Charging_Stations_Near_Work',None,20),'te_cars':('Number_of_Cars_Owned',None,20)}
if XK:
    ORIG=np.sort(pd.read_csv('original.csv').Annual_Income_USD.dropna().unique())
    _v=AL.Annual_Income_USD.values; _j=np.clip(np.searchsorted(ORIG,_v),1,len(ORIG)-1)
    AL['near_orig']=np.where(np.abs(ORIG[_j]-_v)<np.abs(ORIG[_j-1]-_v),ORIG[_j],ORIG[_j-1])
    AL['inc100']=np.floor(AL.Annual_Income_USD/100); AL['inc1000']=np.floor(AL.Annual_Income_USD/1000); AL['com_km']=np.floor(AL.Daily_Commute_km)
    KEYS.update({'te_inc_p1':('Annual_Income_USD',None,1),'te_near':('near_orig',None,1),'te_inc100':('inc100',None,1),'te_inc1000':('inc1000',None,1),'te_com_p1':('Daily_Commute_km',None,1),'te_comkm':('com_km',None,1)})
def kv(c,bw): v=AL[c].values; return v if bw is None else np.floor(v/bw)*bw
KV={k:kv(c,bw) for k,(c,bw,a) in KEYS.items()}
def enc(kvv,idx,g,prior,a): s=pd.Series(kvv[idx]); return ((s.map(g['sum']).fillna(0)+a*prior)/(s.map(g['count']).fillna(0)+a)).values
def stat(kvv,idx,ys): return pd.DataFrame({'k':kvv[idx],'y':ys}).groupby('k').y.agg(['sum','count'])
CLIPALL=(AL.Annual_Income_USD==30000).values
from sklearn.linear_model import LogisticRegression
U=(1.2*AL.Annual_Income_USD/1e5+0.6*AL.Environmental_Concern_Level+2*AL.Subsidy_Available-(AL.Range_Anxiety_Level==1)-3*(AL.Range_Anxiety_Level==2)-5.5).values
lr_=LogisticRegression(C=1e6).fit(U[:n].reshape(-1,1),y); MARGIN=lr_.decision_function(U.reshape(-1,1)); print('recipe-only AUC',A(y,U[:n]),flush=True)
oof=np.zeros(n); tp=np.zeros(nte); teidx=np.arange(n,n+nte)
for k,(tr,va) in enumerate(folds(y,NF,42)):
    if k>=ONLY: break
    prior=y[tr].mean(); D=AL.copy()
    for nm,(c,bw,a) in KEYS.items():
        col=np.full(n+nte,np.nan); kk=KV[nm]
        for a_,b_ in StratifiedKFold(5,shuffle=True,random_state=seed+1).split(tr,y[tr]): col[tr[b_]]=enc(kk,tr[b_],stat(kk,tr[a_],y[tr[a_]]),prior,a)
        gall=stat(kk,tr,y[tr]); col[va]=enc(kk,va,gall,prior,a); col[teidx]=enc(kk,teidx,gall,prior,a)
        if c in ('Annual_Income_USD','near_orig','inc100','inc1000') and (bw is None): col[CLIPALL]=np.nan
        D[nm]=col
    A1=D.iloc[:n].reset_index(drop=True); T1=D.iloc[n:].reset_index(drop=True)
    if name=='lr':
        cols=[c for c in A1.columns if str(A1[c].dtype)!='category']
        Z=pd.concat([A1[cols],T1[cols]],ignore_index=True)
        Z=Z.assign(**{c+'_nan':Z[c].isna().astype(int) for c in cols if Z[c].isna().any()})
        Z['U']=U; Z['U2']=U**2; Z['U3']=U**3; Z['linc']=np.log(Z.Annual_Income_USD)
        D2=pd.get_dummies(pd.concat([A1[['Gender','City_Type','Current_Car_Type']],T1[['Gender','City_Type','Current_Car_Type']]],ignore_index=True),dtype=float)
        Z=pd.concat([Z,D2],axis=1); Zv=SimpleImputer(strategy='median').fit(Z.iloc[:n]).transform(Z); Zv=StandardScaler().fit(Zv[:n]).transform(Zv)
        m=LR2(C=0.05,max_iter=300).fit(Zv[tr],y[tr]); rv=m.decision_function(Zv[va]); rt=m.decision_function(Zv[n:])
        oof[va]=1/(1+np.exp(-rv)); tp+=1/(1+np.exp(-rt))/NF; print(tag,k,A(y[va],oof[va]),flush=True); continue
    if name=='lgb':
        import lightgbm as lgb
        m=lgb.LGBMClassifier(n_estimators=8000,learning_rate=0.03,num_leaves=31,max_depth=5,min_child_samples=50,subsample=0.8,subsample_freq=1,colsample_bytree=0.3,reg_lambda=10,max_bin=MB,verbose=-1,random_state=seed*10+k,n_jobs=8)
        m.fit(A1.iloc[tr],y[tr],init_score=MARGIN[tr],eval_set=[(A1.iloc[va],y[va])],eval_init_score=[MARGIN[va]],eval_metric='auc',callbacks=[lgb.early_stopping(200,verbose=False)])
    else:
        import xgboost as xgb
        m=xgb.XGBClassifier(n_estimators=8000,learning_rate=0.03,max_depth=5,min_child_weight=10,subsample=0.8,colsample_bytree=0.3,reg_lambda=10,max_bin=MB,tree_method='hist',eval_metric='auc',early_stopping_rounds=200,random_state=seed*10+k,n_jobs=8)
        m.fit(A1.iloc[tr],y[tr],base_margin=MARGIN[tr],eval_set=[(A1.iloc[va],y[va])],base_margin_eval_set=[MARGIN[va]],verbose=False)
    if name=='lgb': rv=m.predict(A1.iloc[va],raw_score=True)+MARGIN[va]; rt=m.predict(T1,raw_score=True)+MARGIN[n:]
    else: rv=m.predict(A1.iloc[va],output_margin=True,base_margin=MARGIN[va]); rt=m.predict(T1,output_margin=True,base_margin=MARGIN[n:])
    oof[va]=1/(1+np.exp(-rv)); tp+=1/(1+np.exp(-rt))/NF
    print(tag,k,A(y[va],oof[va]),flush=True)
print(tag,'OOF AUC',A(y,oof),flush=True) if ONLY>=NF else None
if ONLY>=NF: np.save(f'preds/{tag}_oof.npy',oof); np.save(f'preds/{tag}_test.npy',tp)
