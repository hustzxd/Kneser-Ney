# -*- coding:utf-8 -*-
from nltk.util import ngrams
from util.kneser_ney import KneserNeyLM
from util.reader import Reader
import time

train_seg = 'LM_data/train.seg'
lm_arpa = 'train.arpa'

reader = Reader()
sents = reader.read_seg(train_seg)
print('Read sentences done!')

ngrams3 = (ngram for sent in sents for ngram in ngrams(sent, 3, pad_left=False, pad_right=False))

lm = KneserNeyLM(3)
print('Start train...')
start = time.time()
lm.train(ngrams3)
end = time.time()
print('Train cost {:.2f}s'.format(end - start))
lm.save_lm(lm_arpa)
print('Save trained lm to {:s}'.format(lm_arpa))