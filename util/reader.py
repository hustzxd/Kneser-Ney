# -*- coding:utf-8 -*-


class Reader:
    def __init__(self):
        """
        :param filename: train.seg or test.seg
        """

    def read_seg(self, filename):
        sent_list = []
        with open(filename, 'r') as f:
            for line in f:
                line = line.strip('\n').split(' ')
                sent_list.append(line)
        return sent_list



