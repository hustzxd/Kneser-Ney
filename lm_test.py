# -*- coding:utf-8 -*-
from util.reader import Reader
from util.kneser_ney import KneserNeyLM
import time

reader = Reader()
sents = reader.read_seg('LM_data/test.seg')
print('Read sentences done!')

lm = KneserNeyLM(3)

start = time.time()
lm.read_lm('train.arpa')
end = time.time()
print('Load lm done! Cost {:.2f}s'.format(end - start))

print('Test lm begin...')
start = time.time()
lm.test_pp(sents)
end = time.time()
print('Test end. Cost {:.2f}s'.format(end - start))
