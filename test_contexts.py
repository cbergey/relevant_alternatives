import numpy as np
import pandas as pd
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from pytorch_pretrained_bert import BertTokenizer,BertForMaskedLM
import math
import time
import torch
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
import scipy
import csv
from scipy.stats import entropy
from alternative_generator import *

sent1 = "i recommend the candidate for this position . he has great handwriting ."
sent2 = "this student is doing well in kindergarten . he has great handwriting ."
sent3 = "i really enjoyed my date last night . he has great handwriting ."
controlsent = "he has great handwriting ."

'''
print(list(zip(sent1.split(), get_probs(sent1))))
print(list(zip(sent2.split(), get_probs(sent2))))
print(list(zip(sent3.split(), get_probs(sent3))))
print(list(zip(controlsent.split(), get_probs(controlsent))))

print(list(zip(sent1.split(), get_ents(sent1))))
print(list(zip(sent2.split(), get_ents(sent2))))
print(list(zip(sent3.split(), get_ents(sent3))))
print(list(zip(controlsent.split(), get_ents(controlsent))))

print(list(get_n_most_likely(sent1, 11)))
print(list(get_n_most_likely(sent2, 11)))
print(list(get_n_most_likely(sent3, 11)))
print(list(get_n_most_likely(controlsent, 3)))
'''

def get_alt_probs(sentence):
	all_probs = pd.DataFrame()
	for i in range(8, len(sentence.split()) - 1):
		these_probs = pd.DataFrame(list(get_n_most_likely(sentence, i, n = 20)), columns = ['utt','prob'])
		these_probs['slot'] = i
		these_probs['curr_word'] = sentence.split()[i]
		these_probs['rank_alt'] = range(0,20)
		all_probs = all_probs.append(these_probs)
	return(all_probs)

letter_probs = get_alt_probs(sent1)
letter_probs['context'] = "recommendation letter"
kindergarten_probs = get_alt_probs(sent2)
kindergarten_probs['context'] = "kindergarten"
date_probs = get_alt_probs(sent3)
date_probs['context'] = "date"

print(len(letter_probs.index))
print(len(date_probs.index))

plot1 = sns.barplot(data = letter_probs, x = "curr_word", y = "prob", hue = "rank_alt").get_figure()
plot1.savefig("letter_prob_plot.png")

plot2 = sns.barplot(data = kindergarten_probs, x = "curr_word", y = "prob", hue = "rank_alt").get_figure()
plot2.savefig("kindergarten_prob_plot.png")

plot3 = sns.barplot(data = date_probs, x = "curr_word", y = "prob", hue = "rank_alt").get_figure()
plot3.savefig("date_prob_plot.png")
