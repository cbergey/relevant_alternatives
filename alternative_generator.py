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
from scipy.stats import entropy

epsilon = 0.000000000001

def softmax(x, axis=None):
    x = x - x.max(axis=axis, keepdims=True)
    y = np.exp(x)
    return y / y.sum(axis=axis, keepdims=True)

bertMaskedLM = BertForMaskedLM.from_pretrained('bert-base-uncased')
bertMaskedLM.eval()
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def bert_completions(text, model, tokenizer):

  text = '[CLS] ' + text + ' [SEP]'
  tokenized_text = tokenizer.tokenize(text)
  indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
  masked_index = tokenized_text.index('[MASK]')  

  # Create the segments tensors.
  segments_ids = [0] * len(tokenized_text)

  # Convert inputs to PyTorch tensors
  tokens_tensor = torch.tensor([indexed_tokens])
  segments_tensors = torch.tensor([segments_ids])

  # Predict all tokens
  with torch.no_grad():
      predictions = model(tokens_tensor, segments_tensors)
 
  probs = softmax(predictions[0, masked_index].data.numpy())
  words = tokenizer.convert_ids_to_tokens(range(len(probs)))
  word_predictions  = pd.DataFrame({'prob': probs, 'word':words})
  word_predictions = word_predictions.sort_values(by='prob', ascending=False)    
  word_predictions['rank'] = range(word_predictions.shape[0])
  return(word_predictions)
  
def compare_completions(context, candidates, bertMaskedLM, tokenizer):
  continuations = bert_completions(context, bertMaskedLM, tokenizer)
  return(continuations.loc[continuations.word.isin(candidates)])


def bert_score(text, completion, model, tokenizer, normalize = False, return_type = 'prob'):     
  continuations = bert_completions(text, model, tokenizer)
  if not completion in set(continuations.word):
    return(None) # continuation is not in the BERT vocab    
  score = continuations.loc[continuations.word == completion].prob.values[0]
  if return_type == 'normalized_prob':              
      highest_score = continuations.iloc[0].prob
      return( score /  highest_score)
  elif return_type == 'prob':
      return(score)
  elif return_type == 'rank':
      return(np.where(continuations.word == completion)[0][0])
  else:
      raise ValueError('return_type should be "prob" or "rank"')


def get_masked(string, pos):
  string = string.split()
  string[pos] = '[MASK]'
  return(' '.join(string))

def get_ents(utterance):
  positions = []
  for i in range(len(utterance.split())):
    completions = bert_completions(get_masked(utterance, i), bertMaskedLM, tokenizer)
    ent = entropy(completions['prob'])
    positions.append(ent)
  return(positions)

def get_probs(utterance):
  positions = []
  for i in range(len(utterance.split())):
    completions = bert_completions(get_masked(utterance, i), bertMaskedLM, tokenizer)
    if not completions[completions['word'] == utterance.split()[i]].empty:
      word_prob = completions[completions['word'] == utterance.split()[i].lower()]['prob'].iloc[0]
    else:
      word_prob = epsilon
    positions.append(word_prob)
  return(positions)

def get_informativities(probs_arr, ents_arr):
  if len(probs_arr) != len(ents_arr):
    raise Exception("entropy and probability arrays are of different lengths")
  else:
    diff_arr = [-probs_arr[i] - ents_arr[i] for i in range(len(probs_arr))]
    return diff_arr

def get_most_informative(informativities_arr):
  return np.where(informativities_arr == np.amax(informativities_arr))

def get_utterance_perplexity(utterance):
  probs = get_probs(utterance)
  sum = 0.0
  for prob in probs:
    sum = sum + math.log(prob, math.e)
  return math.exp(-sum/len(probs))

def get_utterance_prob(utterance):
  probs = get_probs(utterance)
  sum = 0.0
  for prob in probs:
    sum = sum + math.log(prob, math.e)
  return math.exp(sum)

def replace_word(utterance, pos, word):
  utterance = utterance.split()
  utterance[pos] = word
  return(' '.join(utterance))

def remove_word(utterance, pos):
  utterance = utterance.split()
  del utterance[pos]
  return(' '.join(utterance))

def get_n_most_likely(utterance, word_position, include_null = False, n = 5):
  completions = bert_completions(get_masked(utterance, word_position), bertMaskedLM, tokenizer).head(n)
  completions = completions.assign(sentence = lambda dataframe: dataframe['word'].map(lambda word: replace_word(utterance, word_position, word)))
  return(zip(list(completions['sentence']),list(completions['prob'])))

def add_utt(dict, parent_utt, child_utt, prob = 0):
  if not parent_utt in dict:
    dict[parent_utt] = {}
  dict[parent_utt][child_utt] = prob
  return dict

def normalize_dict(dict, sum):
  for parent_key in dict.items():
    prob_sum = 0.0
    for child_key, prob in child_dict.items():
      prob_sum = prob_sum + prob
  return(dict)

def rec_traverse(utt, depth, dict, iter = 1):
  n = 3
  if iter == depth:
    utt_len = len(utt.split())
    for i in range(utt_len - 1):
      child_utts = get_n_most_likely(utt, i, n = n)
      for child_utt, prob in list(child_utts):
        dict = add_utt(dict, utt, child_utt, prob)
    return dict
  else:
    utt_len = len(utt.split())
    for i in range(utt_len - 1):
      child_utts = get_n_most_likely(utt, i, n = n)
      for child_utt, prob in list(child_utts):
        dict = add_utt(dict, utt, child_utt, prob)
        dict = rec_traverse(child_utt, depth, dict, iter + 1)
    return dict

cat_net = rec_traverse("i love my cat .", 3, {})

def get_nodes_edges(net):
  nodes_dict = {}
  edge_list = []
  iterator = 0
  for parent_key, child_dict in net.items():
    if not parent_key in nodes_dict:
      parent_node = iterator
      nodes_dict[parent_key] = parent_node
      iterator = iterator + 1
    else:
      parent_node = nodes_dict[parent_key]
    for key, val in child_dict.items():
      if not key in nodes_dict:
        child_node = iterator
        nodes_dict[key] = child_node
        iterator = iterator + 1
      else:
        child_node = nodes_dict[key]
      edge_list.append([parent_node, child_node, val])
  return nodes_dict, edge_list

nodes, edges = get_nodes_edges(cat_net)

nodes = pd.DataFrame(nodes.items())
edges = pd.DataFrame(edges)
nodes.to_csv('cat_nodes.csv', index=False, header=False)
edges.to_csv('cat_edges.csv', index=False, header=False)
