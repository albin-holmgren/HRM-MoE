"""Explicit causal-pretraining versus instruction-pair encoding."""
def encode_record(row,tokenizer,max_seq_len,vocab_size=None):
    if row.get('kind')=='pretrain':
        ids=row['token_ids']
        if vocab_size is None:vocab_size=tokenizer.get_vocab_size()
        special={tokenizer.token_to_id(t) for t in ['[PAD]','[UNK]','[BOS]','[SEP]','[EOS]']}
        if not ids or any(type(t) is not int or not 0<=t<vocab_size or t in special for t in ids):
            raise ValueError('Invalid pretraining tokens')
        prefix=[tokenizer.token_to_id('[BOS]')]
        answer=ids+([tokenizer.token_to_id('[EOS]')] if row.get('document_end',True) else [])
    else:
        prefix=[tokenizer.token_to_id('[BOS]')]+tokenizer.encode(row['instruction']).ids+[tokenizer.token_to_id('[SEP]')]
        answer=tokenizer.encode(row['response']).ids+[tokenizer.token_to_id('[EOS]')]
    if len(prefix)+len(answer)-1>max_seq_len:raise ValueError('Sample exceeds max_seq_len; do not silently truncate evidence')
    return prefix,answer
