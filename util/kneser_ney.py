# -*- coding:utf-8 -*-
import math
import random
from collections import Counter, defaultdict


class KneserNeyLM:
    def __init__(self, highest_order, start_pad_symbol='<s>',
                 end_pad_symbol='</s>'):
        """
        Init the language model, just config it.
        Then you can use it to
            train lm using train.seg
            test lm using test.seg
            predict next word.
        :param highest_order :type int n元语言模型
        :param start_pad_symbol
        :param end_pad_symbol
        :return none
        """
        self.highest_order = highest_order
        self.start_pad_symbol = start_pad_symbol
        self.end_pad_symbol = end_pad_symbol
        self.lm = None

    def save_lm(self, filename):
        """
        Save lm to a file.
        :param filename :type string
        :return: none
        """
        with open(filename, 'w') as wr:
            wr.write('\data\\\n')
            wr.write('1-ngram={}\n'.format(len(self.lm[2])))
            wr.write('2-ngram={}\n'.format(len(self.lm[1])))
            wr.write('3-ngram={}\n\n'.format(len(self.lm[0])))
            for i, probs in enumerate(reversed(self.lm)):
                wr.write('\{}-grams:\n'.format(i + 1))
                for key, value in probs.items():
                    if math.fabs(value - 0) < 1e-10:
                        continue
                    wr.write('{:3.6f}\t'.format(math.log(value)))
                    for word in key:
                        wr.write('{:s} '.format(word))
                    wr.write('\n')
                wr.write('\n')
        print('The end')

    def read_lm(self, filename):
        """
        Read lm from a file.
        :param filename: :type(string)
        :return: none
        """
        self.lm = [defaultdict() for i in range(self.highest_order)]
        with open(filename, 'r') as wr:
            head = wr.readline().strip('\n')
            assert head == '\data\\'
            grams = [0 for i in range(self.highest_order)]
            for i in range(self.highest_order):
                grams[i] = wr.readline().strip('\n')
                grams[i] = int(grams[i][grams[i].index('=') + 1:])
            print(grams)
            lines = wr.readlines()
            cnt = 0
            for j, num in enumerate(grams):
                cnt += 2
                order = defaultdict()
                for i in range(num):
                    if len(lines[cnt]) <= 2:
                        print('{:s}'.format(lines[cnt]))
                    prob, ngrams = lines[cnt].strip('\n').split('\t')
                    cnt += 1
                    ngrams = tuple(ngrams.split(' ')[:-1])
                    prob = float(prob)
                    prob = math.exp(prob)
                    order[ngrams] = prob
                self.lm[-j - 1] = order

    def test_pp(self, sents):
        """

        :param sents: :type(list[string])
        :return:
        """
        cross_entropy = 0
        Wt = 0
        for sent in sents:
            prob = self.score_sent(tuple(sent))
            if prob is None:
                with open('except.reg', 'a') as a:
                    for word in sent:
                        a.write('{:s} '.format(word))
                    a.write('\n')
            else:
                cross_entropy += math.log(prob, 2)
                Wt += len(sent)
        cross_entropy = -cross_entropy / Wt
        pp = 2 ** cross_entropy
        print('Cross_entropy: {:.2f} bits/word\n Perplexity: {:.2f}\n'.format(cross_entropy, pp))

    def train(self, ngrams):
        """
        :param ngrams: :type list[tuple(string)] 训练预料按照 n元切分 的list
        :return: none
        """
        kgram_counts = self._calc_adj_counts(Counter(ngrams))
        self.lm = self._calc_probs(kgram_counts)

    # def highest_order_probs(self):
    #     return self.lm[0]

    def _calc_adj_counts(self, highest_order_counts):
        """
        Calculates the adjusted counts for all ngrams up to the highest order.

        :param highest_order_counts :type(dict{tuple->string, int}) Counts of the highest
                order ngrams.
        :return: kgrams_counts :type(list[dict{tuple->string, int}]) List of dict from kgram to counts
                where k is in descending order from highest_order to 0.

        """
        kgrams_counts = [highest_order_counts]
        for i in range(1, self.highest_order):
            last_order = kgrams_counts[-1]
            new_order = defaultdict(int)
            for ngram in last_order.keys():
                prefix = ngram[:-1]
                new_order[prefix] += last_order[ngram]
                if ngram[-1] == self.end_pad_symbol:
                    suffix = ngram[1:]
                    new_order[suffix] += last_order[ngram]
            kgrams_counts.append(new_order)
        return kgrams_counts

    def _calc_probs(self, orders):
        """
        Calculates interpolated probabilities of kgrams for all orders.
        :param orders: :type(list[dict{tuple(string), int}])
        :return:
        """
        backoffs = []
        orders[-1] = self._calc_unigram_probs(orders[-1])  # calculate 1-gram Pkn(wi)
        for order in orders[:-1]:  # 3-grams and 2-grams
            backoff = self._calc_order_backoff_probs(order)
            backoffs.append(backoff)
        backoffs.append(defaultdict(int))
        self._interpolate(orders, backoffs)
        return orders

    def _calc_unigram_probs(self, unigrams):
        """
        P(wi) = c(wi) / sum(c(wi))
        :param unigrams: :type(dict{tuple(string), int}) 一元模型
        :return: :type(dict{tuple(string), float})
        """
        sum_vals = sum(v for v in unigrams.values())
        # unigrams = dict((k, math.log(v / sum_vals)) for k, v in unigrams.items())
        unigrams = dict((k, float(v) / sum_vals) for k, v in unigrams.items())
        return unigrams

    def _calc_order_backoff_probs(self, order):
        """
        Calculate backoff of all order except unigrams.
        :param order: :type(list[dict{tuple(string), int}])
        :return: :type(list[dict{tuple(string), float}])
        """
        num_kgrams_with_count = Counter(
            value for value in order.values() if value <= 4)
        discounts = self._calc_discounts(num_kgrams_with_count)
        prefix_count_sum = defaultdict(list)  # [key 1 cnt, key 2 cnt, key 3+ cnt] 记录(prefix, *) *计数为count次，有sum种
        prefix_sums = defaultdict(int)  # 记录(prefix, *)的总个数，和 2-grams 类似，但不包含(prefix,end)这种情况
        backoffs = defaultdict(float)
        # sum_vals = sum(v for v in order.values())
        for key in order.keys():
            prefix = key[:-1]
            count = order[key]
            if prefix_count_sum.has_key(prefix):
                sums = prefix_count_sum.get(prefix)
                sums[count - 1 if count < 2 else 2] += 1
            else:
                sums = [0 for i in range(3)]
                sums[count - 1 if count < 2 else 2] += 1
                prefix_count_sum[prefix] = sums
            prefix_sums[prefix] += count
            discount = self._get_discount(discounts, count)  # D(c(w i i-n+1))
            order[key] -= discount  # reduction
            # backoffs[prefix] += discount

        for key in order.keys():
            prefix = key[:-1]
            # order[key] = math.log(order[key] / prefix_sums[prefix])
            order[key] /= float(prefix_sums[prefix])
            # order[key] /= float(sum_vals)
            backoffs[prefix] = (self._get_discount(discounts, 1) * prefix_count_sum.get(prefix)[0] +
                                self._get_discount(discounts, 2) * prefix_count_sum.get(prefix)[1] +
                                self._get_discount(discounts, 3) * prefix_count_sum.get(prefix)[2]) \
                               / float(prefix_sums[prefix])
            # for item in prefix_sums:

            # for prefix in backoffs.keys():
            # backoffs[prefix] = math.log(backoffs[prefix] / prefix_sums[prefix])
            # backoffs[prefix] /= float(prefix_sums[prefix])
            # backoffs[prefix] /= float(sum_vals)

        return backoffs

    def _get_discount(self, discounts, count):
        if count > 3:
            return discounts[3]
        return discounts[count]

    def _calc_discounts(self, n):
        """
        Calculate the optimal discount values for kgrams with counts 1, 2, & 3+.
            Y = n1 / (n1 + 2*n2)
                    D1 = 1 - 2Y * n2 / n1
            D(c)=   D2 = 2 - 3Y * n3 / n2
                    D3+ = 3 - 4Y * n4 / n3
        """
        Y = n[1] / float(n[1] + 2 * n[2])
        # Init discounts[0] to 0 so that discounts[i] is for counts of i
        discounts = [0]
        for i in range(1, 4):
            if n[i] == 0:
                discount = 0
            else:
                discount = (i - (i + 1) * Y
                            * n[i + 1] / n[i])
            discounts.append(discount)
        if any(d for d in discounts[1:] if d <= 0):
            raise Exception(
                '***Warning*** Non-positive discounts detected. '
                'Your dataset is probably too small.')
        return discounts

    def _interpolate(self, orders, backoffs):
        """
        Begin interpolate. Calculate higher order using lower order.
        :param orders: :type(list[dict{tuple(string), float}])
        :param backoffs: :type(list[dict{tuple(string), float}])
        :return:
        """
        for last_order, order, backoff in zip(
                reversed(orders), reversed(orders[:-1]), reversed(backoffs[:-1])):
            for kgram in order.keys():
                prefix, suffix = kgram[:-1], kgram[1:]
                # print(last_order[suffix])
                # print(backoff[prefix])
                order[kgram] += last_order[suffix] * backoff[prefix]
        for order in orders:
            for k, v in order.items():
                if math.fabs(v) < 1e-10:
                    del order[k]

    def _prob(self, ngram):
        if len(ngram) == 1:
            order = self.lm[self.highest_order - 1]
            if order.has_key(ngram):
                prob = order[ngram]
                if math.fabs(prob - 0) <= 1e-12:
                    return None
                return prob
            else:
                return None
        order = self.lm[self.highest_order - len(ngram)]
        if order.has_key(ngram):
            return order[ngram]
        else:
            return self._prob(ngram[1:])

    def score_sent(self, sent):
        """
        Return log prob of the sentence.
        :param sent: :type(tuple(string)) example: (<s>, This, is, amazing, </s>,)
        :return:
        """
        sent_prob = 1.0
        if len(sent) < 3:
            return None
        for i in range(self.highest_order - 1):
            prob = self._prob(sent[:i + 1])
            if prob is None or math.fabs(prob - 0) < 1e-10:
                return None
            sent_prob *= prob
        for i in range(len(sent) - self.highest_order + 1):
            ngram = sent[i:i + self.highest_order]
            prob = self._prob(ngram)
            if prob is None or math.fabs(prob - 0) < 1e-10:
                return None
            sent_prob *= prob
        return sent_prob

    def generate_sentence(self, min_length=4):
        """
        Generate a sentence using the probabilities in the language model.
        Params:
            min_length [int] The mimimum number of words in the sentence.
        """
        sent = []
        probs = self.highest_order_probs()
        while len(sent) < min_length + self.highest_order:
            sent = [self.start_pad_symbol] * (self.highest_order - 1)
            # Append first to avoid case where start & end symbal are same
            sent.append(self._generate_next_word(sent, probs))
            while sent[-1] != self.end_pad_symbol:
                sent.append(self._generate_next_word(sent, probs))
        sent = ' '.join(sent[(self.highest_order - 1):-1])
        return sent

    def _get_context(self, sentence):
        """
        Extract context to predict next word from sentence.
        Params:
            sentence [tuple->string] The words currently in sentence.
        """
        return sentence[(len(sentence) - self.highest_order + 1):]

    def _generate_next_word(self, sent, probs):
        context = tuple(self._get_context(sent))
        pos_ngrams = list(
            (ngram, logprob) for ngram, logprob in probs.items()
            if ngram[:-1] == context)
        # Normalize to get conditional probability.
        # Subtract max logprob from all logprobs to avoid underflow.
        _, max_logprob = max(pos_ngrams, key=lambda x: x[1])
        pos_ngrams = list(
            (ngram, math.exp(prob - max_logprob)) for ngram, prob in pos_ngrams)
        total_prob = sum(prob for ngram, prob in pos_ngrams)
        pos_ngrams = list(
            (ngram, prob / total_prob) for ngram, prob in pos_ngrams)
        rand = random.random()
        for ngram, prob in pos_ngrams:
            rand -= prob
            if rand < 0:
                return ngram[-1]
        return ngram[-1]
