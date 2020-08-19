from argparse import ArgumentParser
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import sys
import re

fp = open(sys.argv[1], "r", encoding="utf-8")
lines = fp.read().split("\n")
fp.close()

fp2 = open(sys.argv[1], "r", encoding="utf-8")
lines2 = fp2.read().split("\n")
fp2.close()

def common_prefix(upstream_seq, downstream_seq):
	prefix = ''
	for x,y in zip(upstream_seq, downstream_seq):
		if x == y:
			prefix += x
		else:
			break
	return prefix
prefix_array = []
k = 0
clines = lines2
for i in range(len(lines)-1):
	cols = lines[i].split("\t")
	word = cols[0]
	prev_prefix_length = 0
	prefix_length = 0
	flag = 0
	words = ''
	for j in range(k, len(clines)-1):
		k = k + 1
		words = ''
		#if(flag == 1):
			#break
		cols2 = clines[k].split("\t")
		word2 = cols2[0]
		cp = common_prefix(word, word2)
		#print(cp)
		prefix_length = len(cp)
		print(prefix_length, prev_prefix_length,cp, word, word2)
		if(i ==0):
			prev_prefix_length = prefix_length
			next
		if(prev_prefix_length >= prefix_length and flag == 0 and prefix_length >=3):
			prev_prefix_length = prefix_length
			words += re.sub(cp, '', word2) + ' ' + cols[1]
			prefix_array.append(cp + "{" + words + "},")
			print(cp, prefix_length, word, word2)
			next
		else:
			prev_prefix_length = 0
			prefix_length = 0
			flag = 1

print(prefix_array)