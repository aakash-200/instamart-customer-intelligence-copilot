from pathlib import Path
import sqlite3, pandas as pd, numpy as np
ROOT=Path(__file__).resolve().parents[1]; RAW=ROOT/'data/raw'; CLEAN=ROOT/'data/clean'; OUT=ROOT/'outputs'; CLEAN.mkdir(exist_ok=True);OUT.mkdir(exist_ok=True)
def main():
    c=pd.read_csv(RAW/'customers.csv');p=pd.read_csv(RAW/'products.csv');o=pd.read_csv(RAW/'orders.csv')
    print('RAW SHAPES',c.shape,p.shape,o.shape)
    print('RAW DTYPES customers',c.dtypes.astype(str).to_dict(),'products',p.dtypes.astype(str).to_dict(),'orders',o.dtypes.astype(str).to_dict())
    print('RAW MISSING customers',c.isna().sum().to_dict(),'products',p.isna().sum().to_dict(),'orders',o.isna().sum().to_dict())
    print('AUDIT',{'exact_duplicates':int(o.duplicated().sum()),'pending':int((o.status=='pending').sum()),'distinct_order_id':int(o.order_id.nunique()),'non_numeric_amount':int(pd.to_numeric(o.order_amount,errors='coerce').isna().sum()),'negative_delivery':int((pd.to_numeric(o.delivery_minutes,errors='coerce')<0).sum()),'blank_delivery':int(o.delivery_minutes.isna().sum()),'blank_age':int((c.age.fillna('').astype(str).str.strip()=='').sum()),'age_outside_10_100':int(((pd.to_numeric(c.age,errors='coerce')<10)|(pd.to_numeric(c.age,errors='coerce')>100)).sum()),'raw_city_spellings':int(c.city.nunique()),'raw_gender_spellings':int(c.gender.dropna().nunique()),'blank_gender':int(c.gender.isna().sum())})
    c['city']=c.city.astype(str).str.strip().str.title(); c['gender']=c.gender.fillna('').astype(str).str.strip().str.lower().replace({'m':'male','male':'male','f':'female','female':'female','':'unknown'}); c.loc[~c.gender.isin(['male','female']),'gender']='unknown'
    c['age']=pd.to_numeric(c.age,errors='coerce').astype(float); c.loc[~c.age.between(10,100),'age']=np.nan; c['age']=c.age.fillna(c.groupby('city').age.transform('median')).round().astype(int); c['signup_date']=pd.to_datetime(c.signup_date)
    o['order_ts']=pd.to_datetime(o.order_ts);o=o.drop_duplicates();o=o.sort_values(['order_id','order_ts']).drop_duplicates('order_id',keep='last').copy();o['quantity']=pd.to_numeric(o.quantity);o['discount_pct']=pd.to_numeric(o.discount_pct);o['order_amount']=pd.to_numeric(o.order_amount,errors='coerce');o['delivery_minutes']=pd.to_numeric(o.delivery_minutes,errors='coerce');o.loc[o.delivery_minutes<0,'delivery_minutes']=np.nan
    o=o.merge(p[['product_id','price_inr','category']],on='product_id',how='left'); missing=o.order_amount.isna();o.loc[missing,'order_amount']=(o.loc[missing,'price_inr']*o.loc[missing,'quantity']*(1-o.loc[missing,'discount_pct']/100)).round(2)
    completed=o.status.eq('completed'); med=o.loc[completed].groupby('category').delivery_minutes.transform('median');o.loc[completed,'delivery_minutes']=o.loc[completed,'delivery_minutes'].fillna(med)
    o=o.drop(columns=['price_inr','category']);o['order_ts']=o.order_ts.dt.strftime('%Y-%m-%d %H:%M:%S');c['signup_date']=c.signup_date.dt.strftime('%Y-%m-%d')
    c.to_csv(CLEAN/'customers.csv',index=False);p.to_csv(CLEAN/'products.csv',index=False);o.to_csv(CLEAN/'orders.csv',index=False)
    db=ROOT/'data/instamart.db'
    if db.exists():db.unlink()
    with sqlite3.connect(db) as con:
        c.to_sql('customers',con,index=False);p.to_sql('products',con,index=False);o.to_sql('orders',con,index=False)
    # SQL is run and compactly reported to JSON for reproducibility.
    sql='\n'.join(line for line in (ROOT/'sql/queries.sql').read_text().splitlines() if not line.strip().startswith('--'))
    with sqlite3.connect(db) as con:
        queries=[q.strip() for q in sql.split(';') if q.strip()]
        for i,q in enumerate(queries,1):
            try: df=pd.read_sql_query(q,con); print(f'Q{i}: {len(df)} rows')
            except Exception as e: print(f'Q{i}: ERROR {e}')
        oo=pd.read_sql_query('SELECT o.*,c.city,p.category FROM orders o JOIN customers c USING(customer_id) JOIN products p USING(product_id) WHERE o.status=\'completed\'',con)
        pivot=pd.pivot_table(oo,index='city',columns='category',values='order_amount',aggfunc='sum',fill_value=0,margins=True)
        print('PIVOT GRAND TOTAL',pivot.loc['All','All'])
        print('CITY AGG',oo.groupby('city').order_amount.agg(total='sum',mean='mean',count='count').round(2).to_dict('index'))
        print('AMOUNT STATS',oo.order_amount.mean(),oo.order_amount.median(),oo.order_amount.skew(),oo.order_amount.quantile([.25,.5,.75]).to_dict())
        q1,q3=oo.order_amount.quantile([.25,.75]);iqr=q3-q1;print('IQR FENCES/OUTLIERS',q1-1.5*iqr,q3+1.5*iqr,int((oo.order_amount<q1-1.5*iqr).sum()),int((oo.order_amount>q3+1.5*iqr).sum()))
        print('CORRELATION',oo[['order_amount','discount_pct','quantity','delivery_minutes']].corr().round(4).to_dict())
    assert len(c)==1200 and c.city.nunique()==6 and c.age.isna().sum()==0
    assert c.gender.value_counts().to_dict()=={'female':610,'male':560,'unknown':30}
    assert len(o)==12309 and o.status.value_counts().to_dict()=={'completed':11110,'cancelled':1199}
    assert o.loc[o.status=='completed','order_amount'].sum().round(2)==2998668.80
    assert o.loc[o.status=='completed','delivery_minutes'].isna().sum()==0 and o.loc[o.status=='cancelled','delivery_minutes'].isna().sum()==1199
    print('ALL CHECKPOINTS PASSED')
if __name__=='__main__':main()
