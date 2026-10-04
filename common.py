import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
CATS=['Gender','City_Type','Current_Car_Type','Home_Charging_Possible','Subsidy_Available','Range_Anxiety_Level']
def load(fe=True):
    tr=pd.read_csv('train.csv'); te=pd.read_csv('test.csv')
    y=(tr.Will_Buy_EV=='Yes').astype(int).values
    X=pd.concat([tr.drop(columns=['id','Will_Buy_EV']),te.drop(columns=['id'])],ignore_index=True)
    for c in ['Home_Charging_Possible','Subsidy_Available']: X[c]=(X[c]=='Yes').astype(int)
    X['Range_Anxiety_Level']=X['Range_Anxiety_Level'].map({'Low':0,'Medium':1,'High':2})
    if fe:
        X['commute_clip']=(X.Daily_Commute_km==5.0).astype(int)
        X['income_clip']=(X.Annual_Income_USD==30000).astype(int)
        X['st_sum']=X.Charging_Stations_Near_Home+X.Charging_Stations_Near_Work
        X['st_diff']=X.Charging_Stations_Near_Home-X.Charging_Stations_Near_Work
        X['inc_per_car']=X.Annual_Income_USD/X.Number_of_Cars_Owned
        X['log_inc']=np.log1p(X.Annual_Income_USD)
        X['commute_x_inc']=X.Daily_Commute_km*X.Annual_Income_USD/1e5
        X['sub_x_home']=X.Subsidy_Available*2+X.Home_Charging_Possible
        X['sub_x_env']=X.Subsidy_Available*X.Environmental_Concern_Level
        X['env_x_inc']=X.Environmental_Concern_Level*X.log_inc
        X['sub_city']=X.Subsidy_Available.astype(str)+'_'+X.City_Type
        X['anx_sub']=X.Range_Anxiety_Level*2+X.Subsidy_Available
    for c in ['Gender','City_Type','Current_Car_Type']+(['sub_city'] if fe else []): X[c]=X[c].astype('category')
    n=len(tr); return X.iloc[:n].reset_index(drop=True),y,X.iloc[n:].reset_index(drop=True),te['id'].values
def folds(y,k=5,seed=42): return list(StratifiedKFold(k,shuffle=True,random_state=seed).split(np.zeros(len(y)),y))
