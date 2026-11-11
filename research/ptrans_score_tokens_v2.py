"""Lossless scoring-only fallback for ambiguous unbracketed element pairs."""
from collections import Counter

POLICY='native_first_frozen_vocab_organic_pair_fallback_v2'
ORGANIC_SINGLE=frozenset('BCNOPSFIbcnops')

def compatible_tokens(smiles,vocab,native_tokenize):
 native=native_tokenize(smiles)
 if all(t in vocab for t in native):return native,{}
 result=[];repairs=Counter();unresolved=[]
 for token in native:
  if token in vocab:result.append(token)
  elif len(token)==2 and all(c in ORGANIC_SINGLE and c in vocab for c in token):
   result.extend(token);repairs[token]+=1
  else:unresolved.append(token)
 if unresolved:
  raise ValueError('Candidate has unresolved frozen-vocabulary tokens (no UNK substitution): '+str(sorted(set(unresolved))))
 if ''.join(result)!=smiles:raise AssertionError('Scoring tokenization changed the proposed string')
 return result,dict(repairs)

class ScoringTokenizer:
 def __init__(self,vocab,native):
  self.vocab=vocab;self.native=native;self.candidates=0;self.repaired=0;self.splits=Counter()
 def __call__(self,smiles):
  tokens,splits=compatible_tokens(smiles,self.vocab,self.native)
  self.candidates+=1
  if splits:self.repaired+=1;self.splits.update(splits)
  return tokens
 def metadata(self):
  return {'policy':POLICY,'candidate_occurrences':self.candidates,'repaired_candidate_occurrences':self.repaired,
   'split_ambiguous_tokens':dict(self.splits),'vocabulary_expanded':False,'UNK_substitution':False,
   'candidate_strings_rewritten':0,'bracket_atom_or_ring_tokens_split':False}
