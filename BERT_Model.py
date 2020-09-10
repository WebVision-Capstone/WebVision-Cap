#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 21:29:34 2020

@author: the_squad
"""
#!pip install bert-for-tf2
#!pip install sentencepiece
#!pip install bert

import collections
import json
import pandas as pd
from math import floor
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.models import Model # Keras is the new high level API for TensorFlow
import numpy as np
import os
import regex as re
import bert
from tensorflow.keras.layers import Concatenate
import pickle

#!export CUDA_VISIBLE_DEVICES=0

#physical_devices = tf.config.experimental.list_physical_devices('GPU')
#for physical_device in physical_devices:
#    tf.config.experimental.set_memory_growth(physical_device, True)





#os.chdir('/home/pablo/Desktop/test')
#x = "/home/pablo/Desktop/test/"

os.chdir('/users/paula/scratch/val2')
x = "/users/paula/scratch/val2"
subdirs = [os.path.join(x, o) for o in os.listdir(x) if os.path.isdir(os.path.join(x,o))]

new_dirs=[]
files = []
for dirs in subdirs:
    # r=root, d=directories, f = files
    for r, d, f in os.walk(dirs):
        for file in f:
            files.append(os.path.join(r, file))
            new_dirs.append(os.path.split(dirs)[1])

df = pd.DataFrame(list(zip(files, new_dirs)), columns=['metadata_file','target'])

# To list all files and their full paths (without extensions):
xyz=pd.Series()

for i in np.arange(len(df)):
    df['metadata_file'][i] = df['metadata_file'][i].split(".")[0]

df = df.drop_duplicates().copy()

filelist = list(df.metadata_file)
classlist = list(df.target)

descriptions=[]
ids=[]
targets=[]
pathz=[]
new_dirs=[]
titles=[]

for file in filelist:
    with open(file) as f:
        json_data = json.load(f)
        descriptions.append(json_data['description'])
        ids.append(json_data['id'])
        titles.append(json_data['title'])

new_titles = []
new_descriptions = []

# encode as ascii, decode to convert back to characters from bytes. Also, do regex cleanup on special characters not indicative of meaning
for title in titles:
    new_titles.append(re.sub(r"[^a-zA-Z?.!,¿#@]+", " ",title.encode('ascii',errors='ignore').decode('UTF-8')))
    
for description in descriptions:
    new_descriptions.append(re.sub(r"[^a-zA-Z?.!,¿#@]+", " ",description.encode('ascii',errors='ignore').decode('UTF-8')))

for i in range(len(new_titles)):
    new_titles[i] = new_titles[i].lower()

for i in range(len(new_descriptions)):
    new_descriptions[i] = new_descriptions[i].lower()

for i in range(len(new_titles)):
    new_titles[i] = new_titles[i].lower()

for i in range(len(new_descriptions)):
    new_descriptions[i] = new_descriptions[i].lower()

targets = pd.Series(df['target']).reset_index(drop=True)

# put words into a dictionary for downstream use
def build_dataset(words):
    count = collections.Counter(words).most_common() #.most_common(100) to use the 100 most common words; .most_common() means zero is the most common
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return dictionary, reverse_dictionary

titles_train = new_titles[:floor(len(new_titles)*.75)]
titles_test = new_titles[floor(len(new_titles)*.75):]

descriptions_train = new_descriptions[:floor(len(new_descriptions)*.75)]
descriptions_test = new_descriptions[floor(len(new_descriptions)*.75):]

targets_train = targets[:floor(len(targets)*.75)]
targets_test = targets[floor(len(targets)*.75):]

title_word_list=[]

for i in np.arange(len(new_titles)):
    title_word_list.append(new_titles[i].split())

title_flat_list = []

for sublist in title_word_list:
    for item in sublist:
        title_flat_list.append(item)

title_word_list = title_flat_list.copy()

from collections import Counter

title_word_to_id = Counter()

for word in title_word_list:
    title_word_to_id[word] += 1

# For unique dictionary values of key words
title_word_to_id = {k:(i + 3) for i,(k,v) in enumerate(title_word_to_id.items())}

title_word_to_id["<PAD>"] = 0 # there is no value this replaces; it just adds a pad
title_word_to_id["<START>"] = 1 # BERT doesn't use START tokens so using spaces instead; spaces will be trimmed out
title_word_to_id["<UNK>"] = 2 # UNK tokens are good. BERT converts them to ## so it knows it's unknown
title_word_to_id["<UNUSED>"] = 3

title_id_to_word = {value:key for key, value in title_word_to_id.items()}

description_word_list=[]

for i in np.arange(len(new_descriptions)):
    description_word_list.append(new_descriptions[i].split())

description_flat_list = []

for sublist in description_word_list:
    for item in sublist:
        description_flat_list.append(item)

description_word_list = description_flat_list.copy()

from collections import Counter

description_word_to_id = Counter()

for word in description_word_list:
    description_word_to_id[word] += 1

# For unique dictionary values of key words
description_word_to_id = {k:(i + 3) for i,(k,v) in enumerate(description_word_to_id.items())}

description_word_to_id["<PAD>"] = 0 # there is no value this replaces; it just adds a pad
description_word_to_id["<START>"] = 1 # BERT doesn't use START tokens so using spaces instead; spaces will be trimmed out
description_word_to_id["<UNK>"] = 2 # UNK tokens are good. BERT converts them to ## so it knows it's unknown
description_word_to_id["<UNUSED>"] = 3

description_id_to_word = {value:key for key, value in description_word_to_id.items()}


##################
### BERT MODEL ###
##################

#max_seq_length = 128  # Your choice here.
max_seq_length = 512  # Your choice here.
input_word_ids = tf.keras.layers.Input(shape=(max_seq_length), dtype=tf.int32,
                                       name="input_word_ids")
input_mask = tf.keras.layers.Input(shape=(max_seq_length), dtype=tf.int32,
                                   name="input_mask")
segment_ids = tf.keras.layers.Input(shape=(max_seq_length), dtype=tf.int32,
                                    name="segment_ids")
bert_layer = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/2",
                            trainable=True)
pooled_output, sequence_output = bert_layer([input_word_ids, input_mask, segment_ids])


                                ######################
                                ### BERT TOKENIZER ###
                                ######################


# Set up tokenizer to generate Tensorflow dataset
FullTokenizer = bert.bert_tokenization.FullTokenizer
vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
tokenizer = FullTokenizer(vocab_file, do_lower_case)

train_titles_tokens = list(map(lambda titles_train: ['[CLS]'] + tokenizer.tokenize(titles_train)[:510] + ['[SEP]'], titles_train))
test_titles_tokens = list(map(lambda titles_test: ['[CLS]'] + tokenizer.tokenize(titles_test)[:510] + ['[SEP]'], titles_test))

description_train_tokens = []
for desc_train in descriptions_train:
    desc_train = ['[CLS]'] + tokenizer.tokenize(desc_train)[:510] + ['[SEP]']
    description_train_tokens.append(desc_train)

description_test_tokens = []
for desc_test in descriptions_test:
    desc_test = ['[CLS]'] + tokenizer.tokenize(desc_test)[:510] + ['[SEP]']
    description_test_tokens.append(desc_test)


def get_ids(tokens, tokenizer, max_seq_length):
    """Token ids from Tokenizer vocab"""
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_ids = token_ids + [0] * (max_seq_length-len(token_ids))
    return np.array(input_ids)

def get_masks(tokens, max_seq_length):
    """Mask for padding"""
    if len(tokens)>max_seq_length:
        raise IndexError("Token length more than max seq length!")
    return np.array([1]*len(tokens) + [0] * (max_seq_length - len(tokens)))


def get_segments(tokens, max_seq_length):
    """Segments: 0 for the first sequence, 1 for the second"""
    if len(tokens)>max_seq_length:
        raise IndexError("Token length more than max seq length!")
    segments = []
    current_segment_id = 0
    for token in tokens:
        segments.append(current_segment_id)
        if token == "[SEP]":
            current_segment_id = 1
    return np.array(segments + [0] * (max_seq_length - len(tokens)))

################## Create word ids for BERT
max_seq_length = 512

train_title_tokens_ids = []
test_title_tokens_ids = []

train_description_tokens_ids = []
test_description_tokens_ids = []

for i in np.arange(0, len(train_titles_tokens)):
    this_train_title_token_id = get_ids(train_titles_tokens[i], tokenizer, max_seq_length=max_seq_length)
    train_title_tokens_ids.append(this_train_title_token_id)

for i in np.arange(0, len(test_titles_tokens)):
    this_test_title_token_id = get_ids(test_titles_tokens[i], tokenizer, max_seq_length=max_seq_length)
    test_title_tokens_ids.append(this_test_title_token_id)

for i in np.arange(0, len(description_train_tokens)):
    this_train_description_token_id = get_ids(description_train_tokens[i], tokenizer, max_seq_length=max_seq_length)
    train_description_tokens_ids.append(this_train_description_token_id)

for i in np.arange(0, len(description_test_tokens)):
    this_test_description_token_id = get_ids(description_test_tokens[i], tokenizer, max_seq_length=max_seq_length)
    test_description_tokens_ids.append(this_test_description_token_id)


################## Create text masks for BERT
max_seq_length = 512
    
train_title_tokens_masks = []
test_title_tokens_masks = []

train_description_tokens_masks = []
test_description_tokens_masks = []

for i in np.arange(0, len(train_titles_tokens)):
    this_train_title_token_mask = get_masks(train_titles_tokens[i], max_seq_length=max_seq_length)
    train_title_tokens_masks.append(this_train_title_token_mask)

for i in np.arange(0, len(test_titles_tokens)):
    this_test_title_token_mask = get_masks(test_titles_tokens[i], max_seq_length=max_seq_length)
    test_title_tokens_masks.append(this_test_title_token_mask)

for i in np.arange(0, len(description_train_tokens)):
    this_train_description_token_mask = get_masks(description_train_tokens[i], max_seq_length=max_seq_length)
    train_description_tokens_masks.append(this_train_description_token_mask)

for i in np.arange(0, len(description_test_tokens)):
    this_test_description_token_mask = get_masks(description_test_tokens[i], max_seq_length=max_seq_length)
    test_description_tokens_masks.append(this_test_description_token_mask)

################## Create text segments for BERT

max_seq_length = 512

train_title_tokens_segs = []
test_title_tokens_segs = []

train_description_tokens_segs = []
test_description_tokens_segs = []

input_seg = []
for i in np.arange(0, len(train_titles_tokens)):
    this_train_title_token_seg = get_segments(train_titles_tokens[i], max_seq_length=max_seq_length)
    train_title_tokens_segs.append(this_train_title_token_seg)

for i in np.arange(0, len(test_titles_tokens)):
    this_test_title_token_seg = get_segments(test_titles_tokens[i], max_seq_length=max_seq_length)
    test_title_tokens_segs.append(this_test_title_token_seg)

for i in np.arange(0, len(description_train_tokens)):
    this_train_description_token_seg = get_segments(description_train_tokens[i], max_seq_length=max_seq_length)
    train_description_tokens_segs.append(this_train_description_token_seg)

for i in np.arange(0, len(description_test_tokens)):
    this_test_description_token_seg = get_segments(description_test_tokens[i], max_seq_length=max_seq_length)
    test_description_tokens_segs.append(this_test_description_token_seg)

# Prepping for generator
child_dict1 = dict(zip(ids, train_description_tokens_ids))
child_dict2 = dict(zip(ids, train_description_tokens_masks))
child_dict3 = dict(zip(ids, train_description_tokens_segs))
child_dict4 = dict(zip(ids, train_title_tokens_ids))
child_dict5 = dict(zip(ids, train_title_tokens_masks))
child_dict6 = dict(zip(ids, train_title_tokens_segs))

dict_subset = [child_dict1, child_dict2, child_dict3, child_dict4, child_dict5, child_dict6]
parent_dict = {}
for i in child_dict1.keys():
  parent_dict[i] = tuple(parent_dict[i] for parent_dict in dict_subset)

with open('BERT_VAL_INPUTS_dict.pickle', 'wb') as handle:
    pickle.dump(parent_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    
    
#############################################################################################################
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@#
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
################ Here lies the second iteration of the code, where train gets E2E processing ################
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@#
#############################################################################################################

os.chdir('/users/paula/scratch/train')
x = "/users/paula/scratch/train"

subdirs = [os.path.join(x, o) for o in os.listdir(x) if os.path.isdir(os.path.join(x,o))]

new_dirs=[]
files = []
for dirs in subdirs:
    # r=root, d=directories, f = files
    for r, d, f in os.walk(dirs):
        for file in f:
            files.append(os.path.join(r, file))
            new_dirs.append(os.path.split(dirs)[1])

df = pd.DataFrame(list(zip(files, new_dirs)), columns=['metadata_file','target'])

# To list all files and their full paths (without extensions):
xyz=pd.Series()

for i in np.arange(len(df)):
    df['metadata_file'][i] = df['metadata_file'][i].split(".")[0]

df = df.drop_duplicates().copy()

filelist = list(df.metadata_file)
classlist = list(df.target)

descriptions=[]
ids=[]
targets=[]
pathz=[]
new_dirs=[]
titles=[]

for file in filelist:
    with open(file) as f:
        json_data = json.load(f)
        descriptions.append(json_data['description'])
        ids.append(json_data['id'])
        titles.append(json_data['title'])

new_titles = []
new_descriptions = []

# encode as ascii, decode to convert back to characters from bytes. Also, do regex cleanup on special characters not indicative of meaning
for title in titles:
    new_titles.append(re.sub(r"[^a-zA-Z?.!,¿#@]+", " ",title.encode('ascii',errors='ignore').decode('UTF-8')))
    
for description in descriptions:
    new_descriptions.append(re.sub(r"[^a-zA-Z?.!,¿#@]+", " ",description.encode('ascii',errors='ignore').decode('UTF-8')))

for i in range(len(new_titles)):
    new_titles[i] = new_titles[i].lower()

for i in range(len(new_descriptions)):
    new_descriptions[i] = new_descriptions[i].lower()

for i in range(len(new_titles)):
    new_titles[i] = new_titles[i].lower()

for i in range(len(new_descriptions)):
    new_descriptions[i] = new_descriptions[i].lower()

targets = pd.Series(df['target']).reset_index(drop=True)

# put words into a dictionary for downstream use
def build_dataset(words):
    count = collections.Counter(words).most_common() #.most_common(100) to use the 100 most common words; .most_common() means zero is the most common
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return dictionary, reverse_dictionary

titles_train = new_titles[:floor(len(new_titles)*.75)]
titles_test = new_titles[floor(len(new_titles)*.75):]

descriptions_train = new_descriptions[:floor(len(new_descriptions)*.75)]
descriptions_test = new_descriptions[floor(len(new_descriptions)*.75):]

targets_train = targets[:floor(len(targets)*.75)]
targets_test = targets[floor(len(targets)*.75):]

title_word_list=[]

for i in np.arange(len(new_titles)):
    title_word_list.append(new_titles[i].split())

title_flat_list = []

for sublist in title_word_list:
    for item in sublist:
        title_flat_list.append(item)

title_word_list = title_flat_list.copy()

from collections import Counter

title_word_to_id = Counter()

for word in title_word_list:
    title_word_to_id[word] += 1

# For unique dictionary values of key words
title_word_to_id = {k:(i + 3) for i,(k,v) in enumerate(title_word_to_id.items())}

title_word_to_id["<PAD>"] = 0 # there is no value this replaces; it just adds a pad
title_word_to_id["<START>"] = 1 # BERT doesn't use START tokens so using spaces instead; spaces will be trimmed out
title_word_to_id["<UNK>"] = 2 # UNK tokens are good. BERT converts them to ## so it knows it's unknown
title_word_to_id["<UNUSED>"] = 3

title_id_to_word = {value:key for key, value in title_word_to_id.items()}

description_word_list=[]

for i in np.arange(len(new_descriptions)):
    description_word_list.append(new_descriptions[i].split())

description_flat_list = []

for sublist in description_word_list:
    for item in sublist:
        description_flat_list.append(item)

description_word_list = description_flat_list.copy()

from collections import Counter

description_word_to_id = Counter()

for word in description_word_list:
    description_word_to_id[word] += 1

# For unique dictionary values of key words
description_word_to_id = {k:(i + 3) for i,(k,v) in enumerate(description_word_to_id.items())}

description_word_to_id["<PAD>"] = 0 # there is no value this replaces; it just adds a pad
description_word_to_id["<START>"] = 1 # BERT doesn't use START tokens so using spaces instead; spaces will be trimmed out
description_word_to_id["<UNK>"] = 2 # UNK tokens are good. BERT converts them to ## so it knows it's unknown
description_word_to_id["<UNUSED>"] = 3

description_id_to_word = {value:key for key, value in description_word_to_id.items()}


##################
### BERT MODEL ###
##################

#max_seq_length = 128  # Your choice here.
max_seq_length = 512  # Your choice here.
input_word_ids = tf.keras.layers.Input(shape=(max_seq_length), dtype=tf.int32,
                                       name="input_word_ids")
input_mask = tf.keras.layers.Input(shape=(max_seq_length), dtype=tf.int32,
                                   name="input_mask")
segment_ids = tf.keras.layers.Input(shape=(max_seq_length), dtype=tf.int32,
                                    name="segment_ids")
bert_layer = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/2",
                            trainable=True)
pooled_output, sequence_output = bert_layer([input_word_ids, input_mask, segment_ids])


                                ######################
                                ### BERT TOKENIZER ###
                                ######################


# Set up tokenizer to generate Tensorflow dataset
FullTokenizer = bert.bert_tokenization.FullTokenizer
vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
tokenizer = FullTokenizer(vocab_file, do_lower_case)

train_titles_tokens = list(map(lambda titles_train: ['[CLS]'] + tokenizer.tokenize(titles_train)[:510] + ['[SEP]'], titles_train))
test_titles_tokens = list(map(lambda titles_test: ['[CLS]'] + tokenizer.tokenize(titles_test)[:510] + ['[SEP]'], titles_test))

description_train_tokens = []
for desc_train in descriptions_train:
    desc_train = ['[CLS]'] + tokenizer.tokenize(desc_train)[:510] + ['[SEP]']
    description_train_tokens.append(desc_train)

description_test_tokens = []
for desc_test in descriptions_test:
    desc_test = ['[CLS]'] + tokenizer.tokenize(desc_test)[:510] + ['[SEP]']
    description_test_tokens.append(desc_test)


def get_ids(tokens, tokenizer, max_seq_length):
    """Token ids from Tokenizer vocab"""
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_ids = token_ids + [0] * (max_seq_length-len(token_ids))
    return np.array(input_ids)

def get_masks(tokens, max_seq_length):
    """Mask for padding"""
    if len(tokens)>max_seq_length:
        raise IndexError("Token length more than max seq length!")
    return np.array([1]*len(tokens) + [0] * (max_seq_length - len(tokens)))


def get_segments(tokens, max_seq_length):
    """Segments: 0 for the first sequence, 1 for the second"""
    if len(tokens)>max_seq_length:
        raise IndexError("Token length more than max seq length!")
    segments = []
    current_segment_id = 0
    for token in tokens:
        segments.append(current_segment_id)
        if token == "[SEP]":
            current_segment_id = 1
    return np.array(segments + [0] * (max_seq_length - len(tokens)))

################## Create word ids for BERT
max_seq_length = 512

train_title_tokens_ids = []
test_title_tokens_ids = []

train_description_tokens_ids = []
test_description_tokens_ids = []

for i in np.arange(0, len(train_titles_tokens)):
    this_train_title_token_id = get_ids(train_titles_tokens[i], tokenizer, max_seq_length=max_seq_length)
    train_title_tokens_ids.append(this_train_title_token_id)

for i in np.arange(0, len(test_titles_tokens)):
    this_test_title_token_id = get_ids(test_titles_tokens[i], tokenizer, max_seq_length=max_seq_length)
    test_title_tokens_ids.append(this_test_title_token_id)

for i in np.arange(0, len(description_train_tokens)):
    this_train_description_token_id = get_ids(description_train_tokens[i], tokenizer, max_seq_length=max_seq_length)
    train_description_tokens_ids.append(this_train_description_token_id)

for i in np.arange(0, len(description_test_tokens)):
    this_test_description_token_id = get_ids(description_test_tokens[i], tokenizer, max_seq_length=max_seq_length)
    test_description_tokens_ids.append(this_test_description_token_id)


################## Create text masks for BERT
max_seq_length = 512
    
train_title_tokens_masks = []
test_title_tokens_masks = []

train_description_tokens_masks = []
test_description_tokens_masks = []

for i in np.arange(0, len(train_titles_tokens)):
    this_train_title_token_mask = get_masks(train_titles_tokens[i], max_seq_length=max_seq_length)
    train_title_tokens_masks.append(this_train_title_token_mask)

for i in np.arange(0, len(test_titles_tokens)):
    this_test_title_token_mask = get_masks(test_titles_tokens[i], max_seq_length=max_seq_length)
    test_title_tokens_masks.append(this_test_title_token_mask)

for i in np.arange(0, len(description_train_tokens)):
    this_train_description_token_mask = get_masks(description_train_tokens[i], max_seq_length=max_seq_length)
    train_description_tokens_masks.append(this_train_description_token_mask)

for i in np.arange(0, len(description_test_tokens)):
    this_test_description_token_mask = get_masks(description_test_tokens[i], max_seq_length=max_seq_length)
    test_description_tokens_masks.append(this_test_description_token_mask)

################## Create text segments for BERT

max_seq_length = 512

train_title_tokens_segs = []
test_title_tokens_segs = []

train_description_tokens_segs = []
test_description_tokens_segs = []

input_seg = []
for i in np.arange(0, len(train_titles_tokens)):
    this_train_title_token_seg = get_segments(train_titles_tokens[i], max_seq_length=max_seq_length)
    train_title_tokens_segs.append(this_train_title_token_seg)

for i in np.arange(0, len(test_titles_tokens)):
    this_test_title_token_seg = get_segments(test_titles_tokens[i], max_seq_length=max_seq_length)
    test_title_tokens_segs.append(this_test_title_token_seg)

for i in np.arange(0, len(description_train_tokens)):
    this_train_description_token_seg = get_segments(description_train_tokens[i], max_seq_length=max_seq_length)
    train_description_tokens_segs.append(this_train_description_token_seg)

for i in np.arange(0, len(description_test_tokens)):
    this_test_description_token_seg = get_segments(description_test_tokens[i], max_seq_length=max_seq_length)
    test_description_tokens_segs.append(this_test_description_token_seg)

# Prepping for generator
child_dict1 = dict(zip(ids, train_description_tokens_ids))
child_dict2 = dict(zip(ids, train_description_tokens_masks))
child_dict3 = dict(zip(ids, train_description_tokens_segs))
child_dict4 = dict(zip(ids, train_title_tokens_ids))
child_dict5 = dict(zip(ids, train_title_tokens_masks))
child_dict6 = dict(zip(ids, train_title_tokens_segs))

dict_subset = [child_dict1, child_dict2, child_dict3, child_dict4, child_dict5, child_dict6]
parent_dict = {}
for i in child_dict1.keys():
  parent_dict[i] = tuple(parent_dict[i] for parent_dict in dict_subset)

with open('BERT_TRAIN_INPUTS_dict.pickle', 'wb') as handle:
    pickle.dump(parent_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
# Prepped for generator


#################################################################
#***** Below is a work in progress, not for use right now ******#

# max_seq_length = 500  # Your choice here.
# input_word_ids = tf.keras.layers.Input(shape=(max_seq_length), dtype=tf.int32,
#                                        name="input_word_ids")
# input_mask = tf.keras.layers.Input(shape=(max_seq_length), dtype=tf.int32,
#                                    name="input_mask")
# segment_ids = tf.keras.layers.Input(shape=(max_seq_length), dtype=tf.int32,
#                                     name="segment_ids")
# pooled_output_titles, sequence_output_titles = bert_layer([train_title_tokens_ids, train_title_tokens_masks, train_title_tokens_segs])
# pooled_output_descriptions, sequence_output_descriptions = bert_layer([train_description_tokens_ids, train_description_tokens_masks, train_description_tokens_segs])
#Dense_titles here
#Dense_descriptions here
#Dropout_titles here
#Dropout_descriptions here
#Batch norm the title vectors
#Batch norm the description vectors
#Concat the output
#titles_classifier1 = tf.keras.layers.Dense(5000,activation='sigmoid')(concatenating vectors)

###################combo_pooled_output = Concatenate()([pooled_output_titles, pooled_output_descriptions])
###################combo_sequence_output = Concatenate()([sequence_output_titles, sequence_output_descriptions]) ### Sequence output needs to go somewhere

#titles_classifier1 = tf.keras.layers.Dense(100,activation='sigmoid')(pooled_output_titles)
#titles_classifier2 = tf.keras.layers.Dense(1,activation='sigmoid')(titles_classifier1)

#titles_model = Model(inputs=[train_title_tokens_ids, train_title_tokens_masks, train_title_tokens_segs], outputs=[titles_classifier2])


#descriptions_classifier1 = tf.keras.layers.Dense(100,activation='sigmoid')(pooled_output_descriptions)
#descriptions_classifier2 = tf.keras.layers.Dense(1,activation='sigmoid')(descriptions_classifier1)

#descriptions_model = Model(inputs=[train_description_tokens_ids, train_description_tokens_masks, train_description_tokens_segs], outputs=[descriptions_classifier2])


#### Titles Model Running
#metrics = [tf.keras.metrics.SparseCategoricalAccuracy('accuracy', dtype=tf.float32)]
#loss = tf.keras.losses.BinaryCrossentropy()
#batch_size = 4
#epochs = 5

#model.compile(
#    optimizer='rmsprop',
#    loss=loss,
#    metrics=metrics)

#titles_model.fit(
#     x_data, training_targets,
#     validation_data=(val_data, testing_targets),
#       batch_size=batch_size,
#       epochs=epochs)

# xx = model.predict(x_data)

# #### Descriptions Model Running
# metrics = [tf.keras.metrics.SparseCategoricalAccuracy('accuracy', dtype=tf.float32)]
# loss = tf.keras.losses.BinaryCrossentropy()
# batch_size = 4
# epochs = 5

# descriptions_model.compile(
#     optimizer='rmsprop',
#     loss=loss,
#     metrics=metrics)

# model.fit(
#      x_data, training_targets,
#      validation_data=(val_data, testing_targets),
#       batch_size=batch_size,
#       epochs=epochs)

# xx = model.predict(x_data)

























