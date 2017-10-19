# -*- coding:utf-8 -*-
from util.reader import Reader
from nltk.util import ngrams
from kneser_ney import KneserNeyLM

reader = Reader('LM_data/t.seg')
sents = reader.read()

ss = ngrams(sents[0], 3, pad_left=True, pad_right=True)
gut_ngrams = (ngram for sent in sents for ngram in ngrams(sent, 3, pad_left=True, pad_right=True))

lm = KneserNeyLM(3, gut_ngrams, end_pad_symbol='<s>')