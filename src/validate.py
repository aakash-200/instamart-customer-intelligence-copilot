from pathlib import Path
import sqlite3,pandas as pd
R=Path(__file__).resolve().parents[1]
c=pd.read_csv(R/'data/clean/customers.csv');o=pd.read_csv(R/'data/clean/orders.csv')
checks=[len(c)==1200,c.city.nunique()==6,c.age.isna().sum()==0,c.gender.value_counts().to_dict()=={'female':610,'male':560,'unknown':30},len(o)==12309,o.status.value_counts().to_dict()=={'completed':11110,'cancelled':1199},o.loc[o.status=='completed','order_amount'].sum().round(2)==2998668.80,o.loc[o.status=='completed','delivery_minutes'].isna().sum()==0,o.loc[o.status=='cancelled','delivery_minutes'].isna().sum()==1199]
assert all(checks),checks
with sqlite3.connect(R/'data/instamart.db') as con:assert set(pd.read_sql("select name from sqlite_master where type='table'",con).name)>={'customers','products','orders'}
print('ALL CHECKPOINTS PASSED')
