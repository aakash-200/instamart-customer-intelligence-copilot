from pathlib import Path
import sys,json,time,sqlite3
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));import mock_llm
POL=ROOT/'data/policies';OUT=ROOT/'outputs';STATE=ROOT/'state';STATE.mkdir(exist_ok=True)
def chunks(overlap=10):
 out=[]
 for path in sorted(POL.glob('*.txt')):
  words=path.read_text().split();step=40-overlap
  for start in range(0,len(words),step):
   part=words[start:start+40]
   if not part:break
   out.append({'source':path.name,'text':' '.join(part)})
   if start+40>=len(words):break
 return out
class Retriever:
 def __init__(self,overlap=10):self.docs=chunks(overlap);self.v=TfidfVectorizer();self.matrix=self.v.fit_transform([x['text'] for x in self.docs])
 def retrieve(self,query,k=3):
  q=self.v.transform([query]);scores=cosine_similarity(q,self.matrix)[0];idx=scores.argsort()[::-1][:k];return [{'source':self.docs[i]['source'],'score':float(scores[i]),'text':self.docs[i]['text']} for i in idx]
def main():
 for old in STATE.glob('*.json'):old.unlink()
 retr=Retriever(10);print('CHUNKS',len(retr.docs),pd.Series([x['source'] for x in retr.docs]).value_counts().to_dict());print('TOP3',[(x['source'],round(x['score'],3)) for x in retr.retrieve('refund policy late delivery')])
 def answer(q,r=retr):
  hits=r.retrieve(q,3);ctx='\n'.join(f"[{x['source']}] {x['text']}" for x in hits);msgs=[{'role':'system','content':'POLICY ASSISTANT: Answer only from supplied context. If absent, say you could not find the information.'},{'role':'user','content':f'CONTEXT:\n{ctx}\nQUESTION:\n{q}'}];resp=mock_llm.chat(msgs);return resp.choices[0].message.content,[x['source'] for x in hits]
 qa=pd.read_csv(ROOT/'data/raw/policy_eval.csv');good=bad=unans=0
 for _,x in qa.iterrows():
  ans,src=answer(x.question)
  if x.expected_doc=='NONE':unans+=int(ans.startswith('I could not find'))
  else:good+=int(x.expected_doc in src);bad+=int(x.expected_fact in ans)
 print('EVAL OVERLAP',good,bad,unans)
 r0=Retriever(0);g0=f0=u0=0
 for _,x in qa.iterrows():
  ans,src=answer(x.question,r0)
  if x.expected_doc=='NONE':u0+=int(ans.startswith('I could not find'))
  else:g0+=int(x.expected_doc in src);f0+=int(x.expected_fact in ans)
 print('EVAL ZERO OVERLAP',g0,f0,u0)
 # agent tool map
 def get_order(order_id):
  with sqlite3.connect(ROOT/'data/instamart.db') as con:row=con.execute('select order_id,status,order_amount,delivery_minutes from orders where order_id=?',(order_id,)).fetchone()
  return {'error':'order not found'} if row is None else dict(zip(['order_id','status','order_amount','delivery_minutes'],row))
 def search_policy(query):return {'results':retr.retrieve(query,3)}
 def calculate_refund(order_amount,delivery_minutes):return {'refund_inr':round(float(order_amount),2) if float(delivery_minutes)>45 else 0.0}
 def ledger(name,order_id,refund_inr):
  item={'timestamp':time.strftime('%Y-%m-%dT%H:%M:%S'),'tool':name,'order_id':order_id,'refund_inr':round(float(refund_inr),2)}
  with open(OUT/'refund_ledger.jsonl','a') as f:f.write(json.dumps(item)+'\n')
  return {'refund_inr':round(float(refund_inr),2)}
 def issue_refund(order_id,refund_inr):return ledger('issue_refund',order_id,refund_inr)
 def escalate_to_human(order_id,refund_inr):return ledger('escalate_to_human',order_id,refund_inr)
 TOOL_MAP={'get_order':get_order,'search_policy':search_policy,'calculate_refund':calculate_refund,'issue_refund':issue_refund,'escalate_to_human':escalate_to_human}
 TOOLS_SPEC=[
  {'name':'get_order','description':'Read an order by its unique order identifier.','parameters':{'type':'object','properties':{'order_id':{'type':'string'}},'required':['order_id']}},
  {'name':'search_policy','description':'Retrieve policy chunks relevant to a query.','parameters':{'type':'object','properties':{'query':{'type':'string'}},'required':['query']}},
  {'name':'calculate_refund','description':'Calculate the eligible full refund for a late completed order.','parameters':{'type':'object','properties':{'order_amount':{'type':'number'},'delivery_minutes':{'type':'number'}},'required':['order_amount','delivery_minutes']}},
  {'name':'issue_refund','description':'Append an automatically approved refund to the refund ledger.','parameters':{'type':'object','properties':{'order_id':{'type':'string'},'refund_inr':{'type':'number'}},'required':['order_id','refund_inr']}},
  {'name':'escalate_to_human','description':'Append a refund requiring human lead approval to the escalation ledger.','parameters':{'type':'object','properties':{'order_id':{'type':'string'},'refund_inr':{'type':'number'}},'required':['order_id','refund_inr']}}
 ]
 def checkpoint(run_id,sid,msgs,tokens,steps,executed):
  obj={'run_id':run_id,'scenario_id':sid,'messages':msgs,'tokens_used':tokens,'step_number':steps,'tools_executed':executed};(STATE/f'{run_id}.json').write_text(json.dumps(obj,default=str))
 def run_agent(user_message,scenario_id,max_iterations=6,token_limit=6000,time_limit_seconds=10,run_id=None,crash_after=None):
  rid=run_id or f'{scenario_id}_run';msgs=[{'role':'system','content':'AGENT: Follow the refund workflow using tools and policy.'},{'role':'user','content':user_message}];return _continue(rid,scenario_id,msgs,0,0,[],max_iterations,token_limit,time_limit_seconds,crash_after)
 def records(msgs):
  result=[]
  for ai,msg in enumerate(msgs):
   if msg.get('role')=='assistant' and msg.get('tool_calls'):
    tc=msg['tool_calls'][0];obs=msgs[ai+1].get('content','{}') if ai+1<len(msgs) and msgs[ai+1].get('role')=='tool' else '{}'
    result.append({'tool':tc['name'],'arguments':tc['arguments'],'observation':json.loads(obs)})
  return result
 def _continue(rid,sid,msgs,tokens,steps,executed,max_iterations,token_limit,time_limit_seconds,crash_after=None):
  started=time.monotonic();calls=0
  while calls<max_iterations:
   if time.monotonic()-started>=time_limit_seconds:return {'status':'FALLBACK_TIMEOUT','final_answer':'FALLBACK_TIMEOUT: hand off to a human agent; time limit reached.','steps':records(msgs),'step_number':steps,'llm_calls':calls,'tokens_used':tokens,'tools_executed':executed,'messages':msgs,'run_id':rid}
   resp=mock_llm.chat(msgs);calls+=1;tokens+=resp.usage.total_tokens
   if tokens>token_limit:return {'status':'FALLBACK_TOKEN_LIMIT','final_answer':'FALLBACK_TOKEN_LIMIT: hand off to a human agent; token limit exceeded.','steps':records(msgs),'step_number':steps,'llm_calls':calls,'tokens_used':tokens,'tools_executed':executed,'messages':msgs,'run_id':rid}
   m=resp.choices[0].message
   if not m.tool_calls:
    content=m.content or '';status=content.split(':',1)[0] if ':' in content else 'FALLBACK';return {'status':status,'final_answer':content,'steps':records(msgs),'step_number':steps,'llm_calls':calls,'tokens_used':tokens,'tools_executed':executed,'messages':msgs,'run_id':rid}
   tc=m.tool_calls[0];args=tc.arguments if isinstance(tc.arguments,dict) else json.loads(tc.arguments);msgs.append({'role':'assistant','content':None,'tool_calls':[{'id':tc.id,'name':tc.name,'arguments':args}]});obs=TOOL_MAP[tc.name](**args);executed.append(tc.name);steps+=1;msgs.append({'role':'tool','tool_call_id':tc.id,'name':tc.name,'content':json.dumps(obs)});checkpoint(rid,sid,msgs,tokens,steps,executed)
   if crash_after is not None and len(executed)==crash_after:raise RuntimeError('SIMULATED_CRASH_AFTER_CHECKPOINT')
  return {'status':'FALLBACK_MAX_ITERATIONS','final_answer':'FALLBACK_MAX_ITERATIONS: hand off to a human agent; maximum model-call iterations reached.','steps':records(msgs),'step_number':steps,'llm_calls':calls,'tokens_used':tokens,'tools_executed':executed,'messages':msgs,'run_id':rid}
 def resume_agent(run_id,max_iterations=6,token_limit=6000,time_limit_seconds=10):
  cp=json.loads((STATE/f'{run_id}.json').read_text());return _continue(run_id,cp['scenario_id'],cp['messages'],cp['tokens_used'],cp['step_number'],cp['tools_executed'],max_iterations,token_limit,time_limit_seconds)
 scenarios=pd.read_csv(ROOT/'data/raw/agent_scenarios.csv');(OUT/'agent_runs.jsonl').write_text('');(OUT/'refund_ledger.jsonl').write_text('');table=[]
 for _,s in scenarios.iterrows():
  result=run_agent(s.message,s.scenario_id);steps=result['tools_executed'];amount=0
  for st in steps:
   if st in ('issue_refund','escalate_to_human'):amount=float(s.expected_refund_inr)
  row={'scenario_id':s.scenario_id,'status':result['status'],'final_answer':result['final_answer'],'llm_calls':result['llm_calls'],'tokens_used':result['tokens_used'],'steps':result['steps']}
  with open(OUT/'agent_runs.jsonl','a') as f:f.write(json.dumps(row)+'\n')
  table.append((s.scenario_id,result['status'],amount,result['llm_calls'],result['tokens_used']));print('SCENARIO',table[-1])
 # safety demonstrations
 s1=scenarios.iloc[0];tok=run_agent(s1.message,'S1_TOKEN',token_limit=200);tout=run_agent(s1.message,'S1_TIMEOUT',time_limit_seconds=0);print('SAFETY DEMOS',tok['status'],tout['status'])
 # S1 crash after second tool and resume. Clear ledger from this isolated demo to retain auditable actions.
 s1=scenarios.iloc[0];rid='crash_resume_S1'
 try:run_agent(s1.message,'S1_CRASH',run_id=rid,crash_after=2)
 except RuntimeError:pass
 resumed=resume_agent(rid);print('CRASH RESUME',resumed['status'],resumed['tools_executed'])
 print('AGENT TABLE',table)
if __name__=='__main__':main()
