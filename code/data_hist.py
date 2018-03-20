import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

import re
import numpy as np

def split_by_whitespace(sentence):
    words = []
    for space_separated_fragment in sentence.strip().split():
        words.extend(re.split(" ", space_separated_fragment))
    return [w for w in words if w]


def intstr_to_intlist(string):
    """Given a string e.g. '311 9 1334 635 6192 56 639', returns as a list of integers"""
    return [int(s) for s in string.split()]


context_file = open('../data/train.context')
qn_file = open('../data/train.question')
ans_file = open('../data/train.span')

context_line, qn_line, ans_line = context_file.readline(), qn_file.readline(), ans_file.readline()

context_len = []
qn_len = []
ans_len = []

while context_line and qn_line and ans_line:
  context_tokens = split_by_whitespace(context_line)
  qn_tokens = split_by_whitespace(qn_line)
  ans_span = intstr_to_intlist(ans_line)
  context_line, qn_line, ans_line = context_file.readline(), qn_file.readline(), ans_file.readline()
  assert len(ans_span) == 2
  if ans_span[1] < ans_span[0]:
    print "Found an ill-formed gold span: start=%i end=%i" % (ans_span[0], ans_span[1])
    continue
  context_len.append(len(context_tokens))
  qn_len.append(len(qn_tokens))
  ans_len.append(ans_span[1] - ans_span[0])

context_len = np.array(context_len, dtype = int)
qn_len = np.array(qn_len, dtype = int)
ans_len = np.array(ans_len, dtype = int)

plt.hist(context_len)
plt.xlabel('context length')
plt.ylabel('frequency')
plt.title('histogram of context length')
plt.savefig('context_len.png')
plt.clf()
plt.hist(qn_len)
plt.xlabel('question length')
plt.ylabel('frequency')
plt.title('histogram of question length')
plt.savefig('qn_len.png')
plt.clf()
plt.hist(ans_len)
plt.xlabel('answer length')
plt.ylabel('frequency')
plt.title('histogram of answer length')
plt.savefig('ans_len.png')
plt.clf()
