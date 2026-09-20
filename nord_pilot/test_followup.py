import unittest
from nord_pilot.grounding_check import validate_fixture_answer
from nord_pilot.evaluate_saved import make_counterfactual,scores
class GroundingTests(unittest.TestCase):
    def setUp(self):self.prompt='Source [S777]: Fictional shop Nord0 opens at 12 and closes at 17. Question: When does the shop open?'
    def test_wrong_fact_is_rejected_even_with_right_citation(self):
        self.assertEqual(validate_fixture_answer(self.prompt,'The shop opens at 8. [S777]')['decision'],'abstain')
    def test_correct_fact_uses_actual_source_not_generated_citation(self):
        r=validate_fixture_answer(self.prompt,'The shop opens at 12. [S9]')
        self.assertEqual(r['text'],'The shop opens at 12. [S777]')
    def test_multi_source_or_conflicting_hours_abstain(self):
        for extra in ('Source [S8]: Another shop opens at 10. ', 'Another shop opens at 10. '):
            self.assertEqual(validate_fixture_answer(extra+self.prompt,'The shop opens at 12.')['decision'],'abstain')
    def test_missing_fact_and_wrong_question_abstain(self):
        for prompt in (self.prompt.replace('opens at 12','opening time unknown'),self.prompt.replace('open?','close?')):
            self.assertEqual(validate_fixture_answer(prompt,'The shop opens at 12.')['decision'],'abstain')
    def test_swedish_supported_and_extra_claims_not_repeated(self):
        p='Källa [S999]: Den fiktiva butiken Nord1 öppnar klockan 13 och stänger klockan 17. Fråga: När öppnar butiken?'
        self.assertEqual(validate_fixture_answer(p,'Butiken öppnar klockan 13. All items are free. [S1]')['text'],'Butiken öppnar klockan 13. [S999]')
    def test_counterfactual_changes_both_prompt_and_expected(self):
        r={'id':0,'instruction':self.prompt.replace('opens at 12','opens at 8'),'response':'The shop opens at 8. [S777]'}
        c=make_counterfactual(r)
        self.assertIn('opens at 10',c['instruction']);self.assertIn('opens at 10',c['response'])
        self.assertTrue(scores(c['response'],c['response'])['exact'])
if __name__=='__main__':unittest.main()
