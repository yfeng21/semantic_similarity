- devset.csv : original data
- src_valid.txt	: AMR sequence processed from MR
- valid_tgt_lines.txt : tokenized NL utterance
- valid.txt : tokenized NL utterance along with features extracted from AMR, each feature in the format (beginPos, endPos, featureID)
  - features: ["name", "eatType", "food", "priceRange", "customerrating", "area", "near"], 7 is for punctuation, 8 is for <eos> token
- seg-e2e-60-1-far.txt	: Viterbi segmantation
- gen-e2e-60-1-far.txt	: NL utterance generated from segmentation 

