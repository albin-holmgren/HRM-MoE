"""Self-authored fictional fixture. NOT a language pretraining corpus/benchmark."""
import json
from pathlib import Path
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers
ROOT = Path(__file__).resolve().parent

def main():
    rows = []
    for i in range(160):
        if i % 2:
            prompt = f'Källa [S{i}]: Den fiktiva butiken Nord{i} öppnar klockan {8+i%4} och stänger klockan {17+i%3}. Fråga: När öppnar butiken?'
            answer = f'Butiken öppnar klockan {8+i%4}. [S{i}]'
        else:
            prompt = f'Source [S{i}]: Fictional shop Nord{i} opens at {8+i%4} and closes at {17+i%3}. Question: When does the shop open?'
            answer = f'The shop opens at {8+i%4}. [S{i}]'
        rows.append({'id':i,'instruction':prompt,'response':answer})
    train, valid = rows[:128], rows[128:]
    tok = Tokenizer(models.BPE(unk_token='[UNK]'))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    tok.train_from_iterator((r[k] for r in train for k in ('instruction','response')),
       trainers.BpeTrainer(vocab_size=2048, special_tokens=['[PAD]','[UNK]','[BOS]','[SEP]','[EOS]'], initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    dest = ROOT / 'data'
    dest.mkdir(exist_ok=True)
    tok.save(str(dest/'tokenizer.json'))
    for name, data in [('train',train),('valid',valid)]:
        (dest/f'{name}.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in data))
    print(json.dumps({'train':len(train),'valid':len(valid),'tokenizer_vocab':tok.get_vocab_size(),'purpose':'technical fixture only'}))
if __name__ == '__main__': main()
