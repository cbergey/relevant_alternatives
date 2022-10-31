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

with open('./cat_nodes.csv', newline='') as f:
	reader = csv.reader(f)
	data = list(reader)

sent_dict = {}

for sentence, id in data:
	sent_dict[sentence] = list(zip(get_probs(sentence), get_ents(sentence)))

sent_array = []

for key, val in sent_dict.items():
	for i in range(len(val)):
		print(val[i])
		sent_array.append([key, val[i][0], val[i][1]])

print(sent_array)

sent_df = pd.DataFrame(sent_array, columns = ['sentence', 'prob', 'entropy'])
sns.scatterplot(data=sent_df, x="entropy", y="prob").figure.savefig("prob_ent_plot.png")
