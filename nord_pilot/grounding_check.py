"""Narrow test-only guard for the fictional single-shop opening-hours fixture.
It validates the proposed hour and renders a citation from source metadata.
This is not a general RAG verifier, retriever, or substitute for model evaluation.
"""
import re

def validate_fixture_answer(instruction,generated):
    def abstain(reason):return {'decision':'abstain','text':None,'reason':reason}
    english='Question:' in instruction
    marker='Question:' if english else 'Fråga:'
    if instruction.count(marker)!=1:return abstain('unsupported_question')
    context,question=instruction.split(marker)
    expected_question='When does the shop open?' if english else 'När öppnar butiken?'
    if question.strip()!=expected_question:return abstain('unsupported_question')
    ids=re.findall(r'\[S\d+\]',context)
    if len(ids)!=1:return abstain('ambiguous_or_missing_source')
    pattern=r'opens at (\d{1,2})(?!\d)' if english else r'öppnar klockan (\d{1,2})(?!\d)'
    known=re.findall(pattern,context);proposed=re.findall(pattern,generated)
    if len(known)!=1 or len(proposed)!=1:return abstain('ambiguous_or_missing_fact')
    if not 0<=int(known[0])<=23:return abstain('invalid_hour')
    if int(proposed[0])!=int(known[0]):return abstain('unsupported_answer')
    hour=int(known[0]);prefix='The shop opens at' if english else 'Butiken öppnar klockan'
    return {'decision':'answer','text':f'{prefix} {hour}. {ids[0]}','reason':'hour_matches_single_source; citation_rendered_from_source'}
