from pathlib import Path
import sys,re,json,random,time
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));import mock_llm
EMAIL=re.compile(r'[\w.+-]+@[\w-]+\.[\w.]+');PHONE=re.compile(r'\b[6-9]\d{9}\b');ORDER=re.compile(r'ORD-\d{6}')
CATS={'billing','delivery','technical','fraud'};MAX_INPUT_TOKENS=400;BASE_DELAY=.05
ZERO='CLASSIFY. Allowed categories: billing, delivery, technical, fraud. Reply JSON only: {"category": "...", "reason": "..."}. The category must be one of the four allowed values.'
FEW='CLASSIFY. Allowed categories: billing, delivery, technical, fraud. Reply JSON only: {"category": "...", "reason": "..."}. The category must be one of the four allowed values.\nEXAMPLES:\nTicket: Unauthorized transaction on my card.\n{"category":"fraud","reason":"unauthorized transaction"}\nTicket: My delivery is late.\n{"category":"delivery","reason":"late delivery"}'
def redact(s):
 counts=[]
 for pat,rep in [(EMAIL,'[EMAIL]'),(PHONE,'[PHONE]'),(ORDER,'[ORDER_ID]')]:s,n=pat.subn(rep,s);counts.append(n)
 return s,counts
def call_with_retry(ticket_id,messages):
 attempt=0
 while True:
  try:return mock_llm.chat(messages)
  except mock_llm.TransientLLMError:
   if attempt>=3:raise
   wait=BASE_DELAY*(2**attempt)*random.uniform(.9,1.15)
   print(f'RETRY {ticket_id} retry={attempt} wait={wait:.5f}s')
   time.sleep(wait);attempt+=1
def classify_run(prompt,budget=None,output=None):
 tickets=pd.read_csv(ROOT/'data/raw/tickets.csv',keep_default_na=False);segments=pd.read_csv(ROOT/'outputs/customer_segments.csv');seg=segments.set_index('customer_id').to_dict('index');mock_llm.reset();random.seed(42);rows=[];total=0;classified=0;counts={};correct=0;conf={}
 for _,t in tickets.iterrows():
  tid=t.ticket_id;txt=t['text'];cat='';status='';used=0
  if not txt.strip():status='skipped_empty'
  elif mock_llm.est_tokens(txt)>MAX_INPUT_TOKENS:status='flagged_long'
  else:
   safe,counts_red=redact(txt)
   assert not EMAIL.search(safe) and not PHONE.search(safe) and not ORDER.search(safe)
   if budget is not None and total>=budget:status='budget_exhausted'
   else:
    msgs=[{'role':'system','content':prompt},{'role':'user','content':safe}]
    resp=call_with_retry(tid,msgs)
    used=resp.usage.total_tokens;total+=used;content=resp.choices[0].message.content
    try:
     obj=json.loads(content);candidate=obj.get('category') if isinstance(obj,dict) else None
     if candidate in CATS:cat=candidate;status='classified_json'
     else:status='flagged_invalid_output'
    except Exception:
     words=re.findall(r'[a-z]+',(content or '').lower());found=[w for w in words if w in CATS]
     if len(set(found))==1:cat=found[0];status='classified_keyword'
     else:status='flagged_invalid_output'
    if status.startswith('classified'):
     classified+=1;correct+=int(cat==t.true_category);conf[(t.true_category,cat)]=conf.get((t.true_category,cat),0)+1
  s=seg.get(t.customer_id,{});premium=bool(s.get('is_premium',False));segment_name=s.get('segment_name','')
  if status in {'skipped_empty','flagged_long','flagged_invalid_output','budget_exhausted'}:route='human_review';priority='NORMAL'
  elif cat=='fraud':route='security_team';priority='HIGH'
  elif premium:route='premium_desk';priority='HIGH'
  else:route='support_queue';priority='NORMAL'
  rows.append({'ticket_id':tid,'customer_id':t.customer_id,'status':status,'category':cat,'segment_name':segment_name,'priority':priority,'route':route,'tokens_used':used})
  counts[status]=counts.get(status,0)+1
 df=pd.DataFrame(rows)
 if output:df.to_csv(output,index=False)
 print('SUMMARY',counts,'classified',classified,'accuracy',correct/max(classified,1),'tokens',total,'per_ticket',total/max(classified,1));return df,total,counts,conf

def main():
 tickets=pd.read_csv(ROOT/'data/raw/tickets.csv',keep_default_na=False);tot=[0,0,0]
 for raw_text in tickets.text:
  safe,nums=redact(raw_text)
  for i,n in enumerate(nums):tot[i]+=n
  assert not EMAIL.search(safe) and not PHONE.search(safe) and not ORDER.search(safe)
 print('REDACTION COUNTS email phone order total',tot,sum(tot))
 print('ZERO SHOT');z,zt,zc,zconf=classify_run(ZERO)
 print('ZERO CONFUSION',zconf)
 print('FEW SHOT');f,ft,fc,fconf=classify_run(FEW)
 # demo budget use 6000
 print('BUDGET DEMO');b,bt,bc,_=classify_run(ZERO,6000,ROOT/'outputs/triage_budget_demo.csv');print('BUDGET EXHAUSTED',bc.get('budget_exhausted',0))
 print('FINAL FEW SHOT');final,tt,cc,_=classify_run(FEW,100000,ROOT/'outputs/triage_results.csv')
if __name__=='__main__':main()
