from pathlib import Path
import pandas as pd,numpy as np,joblib
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score,accuracy_score,confusion_matrix,precision_score,recall_score,f1_score
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
ROOT=Path(__file__).resolve().parents[1];O=ROOT/'outputs';O.mkdir(exist_ok=True)
def main():
 c=pd.read_csv(ROOT/'data/clean/customers.csv',parse_dates=['signup_date']);o=pd.read_csv(ROOT/'data/clean/orders.csv',parse_dates=['order_ts']);done=o[o.status=='completed'].copy();snap=pd.Timestamp('2025-10-01')
 rfm=done.groupby('customer_id').agg(last=('order_ts','max'),frequency=('order_id','nunique'),monetary=('order_amount','sum')).reset_index();rfm['recency']=(snap-rfm['last']).dt.days;rfm=rfm.merge(c[['customer_id','city']],on='customer_id');X=rfm[['recency','frequency','monetary']];sc=StandardScaler();Xs=sc.fit_transform(X)
 print('RFM',len(rfm),'recency median/min/max',rfm.recency.median(),rfm.recency.min(),rfm.recency.max(),'frequency mean/max',rfm.frequency.mean(),rfm.frequency.max(),'monetary mean/max',rfm.monetary.mean(),rfm.monetary.max())
 scores=[]
 for k in range(2,9):
  km=KMeans(n_clusters=k,n_init=10,random_state=42);lab=km.fit_predict(Xs);sil=silhouette_score(Xs,lab);scores.append((k,km.inertia_,sil));print('K',k,'inertia',round(km.inertia_,1),'silhouette',round(sil,3))
 k=4;km=KMeans(n_clusters=k,n_init=10,random_state=42);rfm['segment_id']=km.fit_predict(Xs);prof=rfm.groupby('segment_id').agg(size=('customer_id','size'),recency=('recency','mean'),frequency=('frequency','mean'),monetary=('monetary','mean')).sort_values('monetary',ascending=False);print('SEGMENTS\n',prof)
 # Stable semantic names based on cluster profile
 names={};premium=int(prof.index[0])
 most_frequent=int(prof.frequency.idxmax()); most_dormant=int(prof.recency.idxmax())
 for sid,row in prof.iterrows():
  if sid==premium:n='Premium Loyalists'
  elif int(sid)==most_frequent:n='Frequent Essentials'
  elif int(sid)==most_dormant:n='At-Risk Occasional'
  else:n='Developing Customers'
  names[int(sid)]=n
 rfm['segment_name']=rfm.segment_id.map(names);rfm['is_premium']=rfm.segment_id.eq(premium);rfm[['customer_id','recency','frequency','monetary','segment_id','segment_name','is_premium']].to_csv(O/'customer_segments.csv',index=False)
 pca=PCA(n_components=2).fit(Xs);print('PCA',pca.explained_variance_ratio_,pca.explained_variance_ratio_.sum())
 # Build pre-cutoff cohort and label using September only for target.
 cutoff=pd.Timestamp('2025-09-01');pre=o[(o.order_ts<cutoff)];pre_done=pre[pre.status=='completed'];cohort=pre_done.customer_id.unique();sep=done[(done.order_ts>=cutoff)&(done.order_ts<pd.Timestamp('2025-10-01'))].customer_id.unique();
 rows=[]
 for cid in cohort:
  h=pre[pre.customer_id==cid];d=h[h.status=='completed'];last=d.order_ts.max();cat=d.merge(pd.read_csv(ROOT/'data/clean/products.csv'),on='product_id').category.mode().iloc[0]
  rows.append({'customer_id':cid,'recency':(cutoff-last).days,'frequency':d.order_id.nunique(),'monetary':d.order_amount.sum(),'avg_discount':d.discount_pct.mean(),'cancellation_rate':(h.status=='cancelled').sum()/len(h),'avg_delivery':d.delivery_minutes.mean(),'favourite_category':cat})
 f=pd.DataFrame(rows).merge(c,on='customer_id');f['lapsed']=~f.customer_id.isin(sep);f['lapsed']=f.lapsed.astype(int);f['tenure_days']=(cutoff-f.signup_date).dt.days
 print('COHORT',len(f),'lapsed',int(f.lapsed.sum()),round(f.lapsed.mean()*100,2))
 num=['recency','frequency','monetary','avg_discount','cancellation_rate','avg_delivery','age','tenure_days'];cat=['city','gender','favourite_category'];features=num+cat
 X=f[features];y=f.lapsed;assert not X.isna().any().any(), X.isna().sum().to_dict();Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,stratify=y,random_state=42)
 preproc=ColumnTransformer([('num',Pipeline([('imputer',SimpleImputer(strategy='median')),('scale',StandardScaler())]),num),('cat',OneHotEncoder(drop='first',handle_unknown='ignore'),cat)])
 # Baseline
 base=np.zeros(len(yte),dtype=int);print('BASELINE accuracy',accuracy_score(yte,base),'F1',f1_score(yte,base,zero_division=0),'split',len(Xtr),len(Xte))
 log=Pipeline([('prep',preproc),('model',LogisticRegression(max_iter=1000))]);gs1=GridSearchCV(log,{'model__C':[.01,.1,1,10,100]},cv=5,scoring='f1');gs1.fit(Xtr,ytr);bestlog=gs1.best_estimator_;print('LOG',gs1.best_params_,gs1.best_score_)
 print('FREQUENCY SCALER TRAIN MEAN',bestlog.named_steps['prep'].named_transformers_['num'].named_steps['scale'].mean_[1],'FULL COHORT FREQUENCY MEAN',f.frequency.mean())
 rf=Pipeline([('prep',preproc),('model',RandomForestClassifier(n_estimators=100,random_state=42))]);gs2=GridSearchCV(rf,{'model__max_depth':[3,5,8],'model__min_samples_leaf':[5,10,20]},cv=5,scoring='f1');gs2.fit(Xtr,ytr);bestrf=gs2.best_estimator_;print('RF',gs2.best_params_,gs2.best_score_)
 for nm,m in [('LOG',bestlog),('RF',bestrf)]:
  pr=m.predict(Xte);print(nm,'CM',confusion_matrix(yte,pr).tolist(),'P/R/F1',precision_score(yte,pr,zero_division=0),recall_score(yte,pr,zero_division=0),f1_score(yte,pr,zero_division=0))
 names_out=bestrf.named_steps['prep'].get_feature_names_out();imp=bestrf.named_steps['model'].feature_importances_;print('RF TOP FEATURES',sorted(zip(names_out,imp),key=lambda z:z[1],reverse=True)[:5])
 coef=bestlog.named_steps['model'].coef_[0];print('LOG COEFFICIENTS',sorted(zip(bestlog.named_steps['prep'].get_feature_names_out(),coef),key=lambda z:abs(z[1]),reverse=True)[:10])
 # Select model by test F1 for demonstrative persisted model (selection rule documented)
 selected=bestrf if gs2.best_score_>=gs1.best_score_ else bestlog
 # training-only balancing experiment for logistic best C
 from sklearn.utils import resample
 train=pd.concat([Xtr.reset_index(drop=True),ytr.reset_index(drop=True).rename('lapsed')],axis=1);maj=train[train.lapsed==0];minr=train[train.lapsed==1];min_up=resample(minr,replace=True,n_samples=len(maj),random_state=42);bal=pd.concat([maj,min_up]);bal_log=Pipeline([('prep',preproc),('model',LogisticRegression(max_iter=1000,C=gs1.best_params_['model__C']))]);bal_log.fit(bal[features],bal.lapsed);print('BALANCED LOG P/R/F1',precision_score(yte,bal_log.predict(Xte),zero_division=0),recall_score(yte,bal_log.predict(Xte),zero_division=0),f1_score(yte,bal_log.predict(Xte),zero_division=0))
 joblib.dump(selected,O/'lapse_model.joblib');loaded=joblib.load(O/'lapse_model.joblib');assert np.array_equal(selected.predict(Xte),loaded.predict(Xte));print('RELOADED PREDICTIONS IDENTICAL')
 # retention list: post-hoc premium labels from full-window segments
 test_ids=Xte.index;test_frame=f.loc[test_ids];seg=pd.read_csv(O/'customer_segments.csv');premium_ids=set(seg.loc[seg.is_premium,'customer_id']);ix=test_frame[test_frame.customer_id.isin(premium_ids)].copy();ix['lapse_probability']=selected.predict_proba(Xte.loc[ix.index])[:,1];ix['actually_lapsed']=ix.lapsed;ret=ix.sort_values('lapse_probability',ascending=False)[['customer_id','lapse_probability','actually_lapsed']].head(10);ret.to_csv(O/'retention_list.csv',index=False);print('RETENTION truly lapsed',int(ret.actually_lapsed.sum()),'n',len(ret))
if __name__=='__main__':main()
