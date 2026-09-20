"""Independent host watchdog; dry-run by default. Deletes ONLY named VM, retains ALL volumes.
Credentials stay outside the container. Requires explicit --execute when later launched.
"""
import argparse,datetime,json,os,stat,time,urllib.request,urllib.error,uuid
from pathlib import Path
BASE='https://api.verda.com/v1'

def delete_payload(instance):
    uuid.UUID(instance)
    return {'action':'delete','id':[instance],'volume_ids':[],'delete_permanently':False}

def check_identity(info,instance,hostname):
    if info['id']!=instance or info['hostname']!=hostname or not hostname.startswith('nord-hrm-pilot'):
        raise ValueError('Instance identity mismatch; refusing any mutation')
    if not info['instance_type'].startswith('1H100') or info['gpu']['number_of_gpus']!=1:
        raise ValueError('Only one H100 is allowed for this test')

def elapsed_cost(info,now,storage_rate,tax):
    created=datetime.datetime.fromisoformat(info['created_at'].replace('Z','+00:00')).timestamp()
    hours=max(0,(now-created)/3600)
    return hours, hours*(float(info['price_per_hour'])+storage_rate)*(1+tax)

class API:
    def __init__(self,path):
        self.path=path
        if stat.S_IMODE(path.stat().st_mode)&0o077:raise ValueError('Credentials must have mode 600')
    def __call__(self,method,path,payload=None):
        creds=json.loads(self.path.read_text())
        req=urllib.request.Request(BASE+'/oauth2/token',data=json.dumps({'grant_type':'client_credentials','client_id':creds['client_id'],'client_secret':creds['client_secret']}).encode(),headers={'Content-Type':'application/json'})
        with urllib.request.urlopen(req,timeout=20) as r:token=json.load(r)['access_token']
        req=urllib.request.Request(BASE+path,data=json.dumps(payload).encode() if payload is not None else None,method=method,headers={'Authorization':'Bearer '+token,'Content-Type':'application/json'})
        try:
            with urllib.request.urlopen(req,timeout=20) as r:
                data=r.read()
                if not data:return None
                try:return json.loads(data)
                except json.JSONDecodeError:
                    value=data.decode().strip()
                    uuid.UUID(value)  # Creation endpoints return an unquoted UUID.
                    return value
        except urllib.error.HTTPError as e:
            if method=='GET' and e.code==404:return None
            raise RuntimeError(f'Verda API returned HTTP {e.code}') from None

def remove_instance(api,instance):
    result=api('PUT','/instances',delete_payload(instance))
    if result is not None:
        if not isinstance(result,list) or not any(x.get('instanceId')==instance and x.get('status')=='success' for x in result):
            raise RuntimeError('Provider did not confirm delete request')
    for _ in range(20):
        info=api('GET','/instances/'+instance)
        if info is None or info.get('status') in ('discontinued','notfound'):
            return
        time.sleep(3)
    raise RuntimeError('Deletion not yet confirmed; operator must check console')

def main():
    p=argparse.ArgumentParser();p.add_argument('--instance',required=True);p.add_argument('--hostname',required=True)
    p.add_argument('--credentials',type=Path);p.add_argument('--state-dir',type=Path,required=True)
    p.add_argument('--storage-hourly',type=float,required=True);p.add_argument('--tax-fraction',type=float,required=True)
    p.add_argument('--max-hours',type=float,default=2);p.add_argument('--stop-usd',type=float,default=20)
    p.add_argument('--execute',action='store_true');a=p.parse_args()
    payload=delete_payload(a.instance)
    if a.tax_fraction<0 or a.storage_hourly<0:raise ValueError('Rates must be nonnegative')
    if not 0<a.max_hours<=2 or not 0<a.stop_usd<=20:raise ValueError('Guard bounds may only tighten the original limits')
    if not a.execute:
        print(json.dumps({'mode':'dry-run','api_request':payload,'delete_at_hours_since_creation':a.max_hours,'stop_at_estimated_usd':a.stop_usd,'hard_cash_target':25,'retained_volumes_still_billed':True}));return
    if a.credentials is None:raise ValueError('A private credentials file is required')
    api=API(a.credentials);info=api('GET','/instances/'+a.instance)
    if info is None:raise ValueError('Instance does not exist')
    check_identity(info,a.instance,a.hostname)
    a.state_dir.mkdir(parents=True,exist_ok=True)
    (a.state_dir/'armed.json').write_text(json.dumps({'instance':a.instance,'pid':os.getpid(),'armed_at':time.time()}))
    while True:
        hours,cost=elapsed_cost(info,time.time(),a.storage_hourly,a.tax_fraction)
        (a.state_dir/'status.json').write_text(json.dumps({'hours':hours,'estimated_usd':cost,'checked_at':time.time()}))
        if hours>=a.max_hours or cost>=a.stop_usd or float(info['price_per_hour'])>3.5 or (a.state_dir/'finished').exists():break
        time.sleep(15)
        try:
            fresh=api('GET','/instances/'+a.instance)
            if fresh is None or fresh.get('status') in ('discontinued','notfound'):return
            check_identity(fresh,a.instance,a.hostname);info=fresh
        except Exception as e:
            print('Provider check failed; attempting early cleanup:',type(e).__name__,flush=True)
            break
    for attempt in range(10):
        try:
            remove_instance(api,a.instance)
            (a.state_dir/'deleted.json').write_text(json.dumps({'instance':a.instance,'volumes_retained':True,'time':time.time()}));return
        except Exception as e:
            print('Cleanup not confirmed:',type(e).__name__,'attempt',attempt+1,flush=True);time.sleep(10)
    raise RuntimeError('BILLING STOP NOT CONFIRMED. Operator must delete VM with no volumes selected.')
if __name__=='__main__':main()
