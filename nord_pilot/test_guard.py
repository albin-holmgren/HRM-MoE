import datetime,unittest
from nord_pilot.verda_guard import delete_payload,check_identity,elapsed_cost,remove_instance
ID='11111111-2222-4333-8444-555555555555'
class GuardTests(unittest.TestCase):
    def test_delete_keeps_all_volumes(self):
        self.assertEqual(delete_payload(ID)['volume_ids'],[])
        self.assertFalse(delete_payload(ID)['delete_permanently'])
    def test_wrong_target_rejected(self):
        d={'id':ID,'hostname':'production','instance_type':'1H100.80S.30V','gpu':{'number_of_gpus':1}}
        with self.assertRaises(ValueError):check_identity(d,ID,'production')
    def test_setup_time_counts(self):
        d={'created_at':'2026-09-20T00:00:00Z','price_per_hour':3.35}
        now=datetime.datetime(2026,9,20,2,tzinfo=datetime.timezone.utc).timestamp()
        hours,cost=elapsed_cost(d,now,0.05,0.25)
        self.assertEqual(hours,2);self.assertAlmostEqual(cost,8.5)
    def test_delete_failure_is_not_success(self):
        with self.assertRaises(RuntimeError):remove_instance(lambda *args:[{'instanceId':ID,'status':'error'}],ID)
    def test_delete_confirmed_by_get(self):
        calls=[]
        def api(method,path,payload=None):
            calls.append((method,path,payload))
            return [{'instanceId':ID,'status':'success'}] if method=='PUT' else None
        remove_instance(api,ID)
        self.assertEqual([x[0] for x in calls],['PUT','GET'])
if __name__=='__main__':unittest.main()
