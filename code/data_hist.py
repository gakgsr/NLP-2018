import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

import numpy as np

def split_by_whitespace(sentence):
    words = []
    for space_separated_fragment in sentence.strip().split():
        words.extend(re.split(" ", space_separated_fragment))
    return [w for w in words if w]


def intstr_to_intlist(string):
    """Given a string e.g. '311 9 1334 635 6192 56 639', returns as a list of integers"""
    return [int(s) for s in string.split()]


def sentence_to_token_ids(sentence, word2id):
    """Turns an already-tokenized sentence string into word indices
    e.g. "i do n't know" -> [9, 32, 16, 96]
    Note any token that isn't in the word2id mapping gets mapped to the id for UNK
    """
    tokens = split_by_whitespace(sentence) # list of strings
    ids = [word2id.get(w, UNK_ID) for w in tokens]
    return tokens, ids


context_file = open('../data/train.context')
qn_file = open('../data/train.question')
ans_file = open('../data/train.answer')

context_line, qn_line, ans_line = context_file.readline(), qn_file.readline(), ans_file.readline()

context_len = []
qn_len = []
ans_len = []

while context_line and qn_line and ans_line:
  context_tokens, context_ids = sentence_to_token_ids(context_line, word2id)
  qn_tokens, qn_ids = sentence_to_token_ids(qn_line, word2id)
  ans_span = intstr_to_intlist(ans_line)
  context_line, qn_line, ans_line = context_file.readline(), qn_file.readline(), ans_file.readline()
  assert len(ans_span) == 2
  if ans_span[1] < ans_span[0]:
    print "Found an ill-formed gold span: start=%i end=%i" % (ans_span[0], ans_span[1])
    continue
  context_len.append(len(context_tokens))
  qn_len.append(len(context_tokens))
  ans_len.append(ans_span[1] - ans_span[0])

context_len = np.array(context_len, dtype = int)
qn_len = np.array(qn_len, dtype = int)
ans_len = np.array(ans_len, dtype = int)

plt.hist(context_len)
plt.savefig('context_len.jpg')
plt.clf()
plt.hist(qn_len)
plt.savefig('qn_len.jpg')
plt.clf()
plt.hist(ans_len)
plt.savefig('ans_len.jpg')
plt.clf()