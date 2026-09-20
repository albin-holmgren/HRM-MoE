import unittest
from tokenizers import Tokenizer,models,trainers,pre_tokenizers
from nord_pilot.data_tools.prepare_real import curate,split_for,reject_reason
from nord_pilot.data_tools.records import encode_record

class DataTests(unittest.TestCase):
    def row(self,text,group='example.org'):
        return dict(text=text,doc_id=group,group=group,source='test')
    def text(self):
        return ' '.join('The educational chapter number '+str(i)+f' explains method{i} topic{i} question{i} through observation and experimentation.' for i in range(40))
    def test_exact_and_near_duplicates_removed_before_split(self):
        t=self.text();r,d=curate([self.row(t),self.row(t.upper(),'a.org'),self.row(t+' A brief new ending introduces another concept.','b.org')])
        self.assertEqual(len(r),1);self.assertEqual(d['exact_duplicate'],1);self.assertEqual(d['near_duplicate'],1)
    def test_group_split_stable_and_all_splits_present(self):
        groups=[str(i)+'.org' for i in range(1000)]
        self.assertEqual({split_for(g) for g in groups},{'train','valid','test'})
        self.assertTrue(all(split_for(g)==split_for(g) for g in groups))
    def test_low_quality_and_benchmark_name_rejected(self):
        for suffix,reason in [(' Contact me at person@example.com','email_present'),(' The GSM8K benchmark','benchmark_name'),(' [EOS]','control_token_text')]:
            self.assertEqual(reject_reason(self.row(self.text()+suffix)),reason)
    def test_causal_pretraining_has_no_future_prefix(self):
        tok=Tokenizer(models.BPE(unk_token='[UNK]'));tok.pre_tokenizer=pre_tokenizers.ByteLevel();tok.train_from_iterator([self.text()],trainers.BpeTrainer(vocab_size=512,special_tokens=['[PAD]','[UNK]','[BOS]','[SEP]','[EOS]'],initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),show_progress=False))
        ids=tok.encode('This is text.').ids
        p,a=encode_record({'kind':'pretrain','token_ids':ids},tok,256)
        self.assertEqual(p,[tok.token_to_id('[BOS]')]);self.assertEqual(a,ids+[tok.token_to_id('[EOS]')])
        self.assertEqual(encode_record({'kind':'pretrain','token_ids':ids,'document_end':False},tok,256)[1],ids)
        with self.assertRaises(ValueError):encode_record({'kind':'pretrain','token_ids':[tok.token_to_id('[EOS]')]},tok,256)
        with self.assertRaises(ValueError):encode_record({'kind':'pretrain','token_ids':ids},tok,2)
if __name__=='__main__':unittest.main()
