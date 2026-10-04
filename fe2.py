import numpy as np,pandas as pd
from common import load
def add_art(X,T,which=('dig','cnt','mod','com')):
    A=pd.concat([X,T],ignore_index=True); inc=A.Annual_Income_USD.astype(np.int64)
    if 'dig' in which:
        for i,nm in enumerate(['ones','tens','hund','thou','tthou','hthou']): A['inc_'+nm]=(inc//10**i)%10
    if 'mod' in which:
        A['inc_m100']=inc%100; A['inc_m1000']=inc%1000; A['inc_r10']=(inc%10==0).astype(int); A['inc_r100']=(inc%100==0).astype(int)
    if 'cnt' in which:
        for c in ['Annual_Income_USD','Daily_Commute_km','Age','Charging_Stations_Near_Home','Charging_Stations_Near_Work']:
            A[c+'_cnt']=A[c].map(A[c].value_counts())
        A['inc_cnt_log']=np.log1p(A.Annual_Income_USD_cnt)
    if 'com' in which:
        d=A.Daily_Commute_km; A['com_dec']=(d*10).round().astype(int)%10; A['com_int']=(d%1==0).astype(int); A['com_frac']=d%1
    n=len(X); return A.iloc[:n].reset_index(drop=True),A.iloc[n:].reset_index(drop=True)
