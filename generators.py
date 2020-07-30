"""Data generators
"""

import glob
import os
import json
from pathlib import Path
from typing import Tuple, List, Union

from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import Sequence

def encode_sentence(s: str, tokenizer) -> np.ndarray:
    """Encode a sentence with bert tokens

    :param s: sentence
    :param tokenizer: the BERT tokenizer
    """
    tokens = [i for i in list(tokenizer.tokenize(s))]
    tokens = tokens[1:]
    return np.array(tokenizer.convert_tokens_to_ids(tokens))

def bert_encode(x_data: List[str], tokenizer, seq_len: int) -> List[tf.Tensor]:
    """Encode text data with BERT tokens

    :param x_data: list of strings for conversion
    :param tokenizer: the BERT tokenizer
    :param seq_len: length of sequences
    """
    
    data = list()
    for i in x_data:
        tmp = encode_sentence(i, tokenizer)
        if len(tmp) > seq_len-1:
            tmp = tmp[0:seq_len-1]
        data.append(tmp)
    sentence = tf.ragged.constant(data)
    
    cls = [tokenizer.convert_tokens_to_ids(['[CLS]'])]*sentence.shape[0]
    input_word_ids = tf.concat([cls, sentence], axis=-1)
    input_mask = tf.ones_like(input_word_ids).to_tensor()

    type_cls = tf.zeros_like(cls)
    type_s1 = tf.zeros_like(sentence)
    input_type_ids = tf.concat(
        [type_cls, type_s1], axis=-1).to_tensor()

    inputs = [
        input_word_ids.to_tensor(),
        input_mask,
        input_type_ids]

    return inputs

def format_metadata_output(descriptions: List[str],
                           titles: List[str], 
                           tokenizer = None, 
                           max_len: int = 0) -> List[Union[np.ndarray, tf.Tensor]]:
    """Format metadata output for BERT and USE models
    """
    formatted_data = list()

    if tokenizer is not None:
        # encode descriptions
        formatted_data += bert_encode(descriptions, tokenizer, max_len)
        # encode titles
        formatted_data += bert_encode(titles, tokenizer, max_len)
    else:
        formatted_data += descriptions
        formatted_data += titles
    return formatted_data


class BaseDataGenerator(Sequence):
    """Keras-like data generator for image and metadata
    """
    
    def __init__(self,
                 path: str,
                 batch_size: int,
                 target_size: Tuple[int, int],
                 scaling: float = (1. / 255),
                 sample_frac: float = None,
                 preshuffle: bool = True,
                 class_limit: int = None) -> None:
        """Constructor
        
        :param path: path to data tree
        :param batch_size: size of batches used in training
        :param target_size: image size
        :param scaling: image scaling value
        :param sample_frac: per class sampling
        """
        # arguments
        self.path = path
        self.batch_size = batch_size
        self.target_size = target_size
        self.scaling = scaling
        self.sample_frac = sample_frac
        # derived
        self.files = list()
        self.json_to_image_mapping = dict()
        self.label_mapping = dict()
        self.classes = list()

        # create a label maker
        # targeting sparse categorical crossentropy
        classes = sorted(glob.glob(self.path + '*')[ : class_limit ])
        for i, label in enumerate(classes):
            label = label.split('/')[-1]
            self.classes.append(label)
            self.label_mapping[label] = i
        
        # get the file pathes
        # sample on a per class basis
        for class_path in classes:
            class_files = glob.glob(class_path + '/*.*')
            if self.sample_frac is not None:
                class_files = np.random.choice(
                    class_files,
                    np.floor(len(class_files) * self.sample_frac).astype(int),
                    replace = False
                    )
            self.files += list(class_files)
        
        # preshuffle for first epoch
        if preshuffle:
            self.on_epoch_end()
        
        print(f"Found {len(self.files)} instances belonging to {len(self.classes)} classes")

    @property
    def num_classes(self) -> int:
        """The number of classes found
        """
        return len(self.classes)

    @property
    def num_instances(self) -> int:
        """The number of classes found
        """
        return len(self.files)

    @staticmethod
    def _load_metadata(path: str, lower_case: bool = False) -> Tuple[str, str]:
        """Load metadata from the JSON files
        """
        with open(Path(path).with_suffix(''), 'r') as in_file:
            items = json.load(in_file)

            desc = items['description']
            titl = items['title']

            if lower_case:
                desc = desc.lower()
                titl = titl.lower()

            return desc, titl

    def __len__(self) -> int:
        """Denotes the number of batches per epoch
        """
        return int(np.floor(len(self.files) / self.batch_size))

    def on_epoch_end(self) -> None:
        """Updates indexes after each epoch
        """
        np.random.shuffle(self.files)

class ImageMetaDataGenerator(BaseDataGenerator):
    """Keras-like data generator for image and metadata
    """
    
    def __init__(self,
                 path: str,
                 batch_size: int,
                 target_size: Tuple[int, int],
                 scaling: float = (1. / 255),
                 sample_frac: float = None,
                 class_limit: int = None,
                 bert_tokenizer = None,
                 bert_sentence_len: int = 0) -> None:
        """Constructor
        
        :param path: path to data tree
        :param batch_size: size of batches used in training
        :param target_size: image size
        :param scaling: image scaling value
        :param sample_frac: per class sampling
        :param preshuffle: shuffle before first epoch
        :param class_limit: a limit on the number of classes used for training
        :param bert_tokenizer: the BERT tokenizer (if using BERT)
        :param bert_sentence_len: the size of the sentences lengths
        """
        super().__init__(path,
                         batch_size,
                         target_size,
                         scaling,
                         sample_frac,
                         class_limit=class_limit)
    
        # set up BERT items if using BERT
        self.tokenizer = bert_tokenizer
        self.max_seq_len = bert_sentence_len

        if self.tokenizer is not None:
             self.bert_tokenize = True
        else:
             self.bert_tokenize = False

    def _load_image(self, path: str) -> np.ndarray:
        """Load image convert to RGB and resize
        """
        img = Image.open(path)
        img = img.resize(self.target_size).convert('RGB')
        img = np.array(img) * self.scaling
        return img
    
    def __getitem__(self, index: int) -> Tuple[List[np.ndarray], np.ndarray]:
        """Generate one batch of data

        Returned as ([Image, Description, Title], label)
        """
        labels = list()
        images = list()
        descriptions = list()
        titles = list()
        
        # get a batch
        f_batch = self.files[index * self.batch_size:(index + 1) * self.batch_size]
        
        # prepare the batch
        for f in f_batch:
            # get the text data
            desc, titl = self._load_metadata(f, self.bert_tokenize)
            descriptions.append(desc)
            titles.append(titl)
            # get the image
            images.append(
                self._load_image(f)
            )
            # get the label
            labels.append(
                self.label_mapping[
                    f.split('/')[-2]
                ]
            )

        formatted_data = format_metadata_output(
            descriptions,
            titles,
            self.tokenizer,
            self.max_seq_len
        )

        img_reshape_size = [len(f_batch)] + list(self.target_size) + [3]
        images = np.array(images).reshape(*img_reshape_size)
        
        formatted_data = [images] + formatted_data

        # images, descriptions, titles, labels
        return (formatted_data, np.array(labels))

class MetaDataGenerator(BaseDataGenerator):
    """Keras-like data generator for metadata
    """
    
    def __init__(self,
                 path: str,
                 batch_size: int,
                 target_size: Tuple[int, int],
                 scaling: float = (1. / 255),
                 sample_frac: float = None,
                 preshuffle: bool = True,
                 class_limit: int = None,
                 bert_tokenizer = None,
                 bert_sentence_len: int = 0
                 ) -> None:
        """Constructor
        
        :param path: path to data tree
        :param batch_size: size of batches used in training
        :param target_size: image size
        :param scaling: image scaling value
        :param sample_frac: per class sampling
        :param preshuffle: shuffle before first epoch
        :param class_limit: a limit on the number of classes used for training
        :param bert_tokenizer: the BERT tokenizer (if using BERT)
        :param bert_sentence_len: the size of the sentences lengths
        """
        super().__init__(path,
                         batch_size,
                         (300, 300),
                         1. / 255,
                         sample_frac,
                         preshuffle,
                         class_limit=class_limit)

        # set up BERT items if using BERT
        self.tokenizer = bert_tokenizer
        self.max_seq_len = bert_sentence_len

        if self.tokenizer is not None:
             self.bert_tokenize = True
        else:
             self.bert_tokenize = False
    
    def __getitem__(self, index: int) -> Tuple[List[Union[np.ndarray, tf.Tensor]], np.ndarray]:
        """Generate one batch of data

        (Returned as [Description, Title], label)
        """
        labels = list()
        descriptions = list()
        titles = list()
        
        # get a batch
        f_batch = self.files[index * self.batch_size:(index + 1) * self.batch_size]
        
        # prepare the batch
        for f in f_batch:
            # get the text data
            desc, titl = self._load_metadata(f, self.bert_tokenize)
            descriptions.append(desc)
            titles.append(titl)
            # get the label
            labels.append(
                self.label_mapping[
                    f.split('/')[-2]
                ]
            )
        
        formatted_data = format_metadata_output(
            descriptions,
            titles,
            self.tokenizer,
            self.max_seq_len
        )

        # descriptions, titles, labels
        return (formatted_data, np.array(labels))

class ImageGenerator(BaseDataGenerator):
    """Keras-like data generator for image and metadata
    """
    
    def __init__(self,
                 path: str,
                 batch_size: int,
                 target_size: Tuple[int, int],
                 scaling: float = (1. / 255),
                 sample_frac: float = None,
                 class_limit: int = None) -> None:
        """Constructor
        
        :param path: path to data tree
        :param batch_size: size of batches used in training
        :param target_size: image size
        :param scaling: image scaling value
        :param sample_frac: per class sampling
        """
        super().__init__(path,
                         batch_size,
                         target_size,
                         scaling,
                         sample_frac,
                         class_limit=class_limit)
    
    def _load_image(self, path: str) -> np.ndarray:
        """Load image convert to RGB and resize
        """
        img = Image.open(path)
        img = img.resize(self.target_size).convert('RGB')
        img = np.array(img) * self.scaling
        return img
    
    def __getitem__(self, index: int) -> Tuple[List[np.ndarray], np.ndarray]:
        """Generate one batch of data

        Returned as (Image, label)
        """
        labels = list()
        images = list()
        
        # get a batch
        f_batch = self.files[index * self.batch_size:(index + 1) * self.batch_size]
        
        # prepare the batch
        for f in f_batch:
            # get the image
            images.append(
                self._load_image(f)
            )
            # get the label
            labels.append(
                self.label_mapping[
                    f.split('/')[-2]
                ]
            )

        img_reshape_size = [len(f_batch)] + list(self.target_size) + [3]
        
        # images, descriptions, titles, labels
        return (np.array(images).reshape(*img_reshape_size), #X
               np.array(labels)) #y

def get_generators(generator_type: str,
                   primary_path: str,
                   batch_size: int,
                   img_size: Tuple[int],
                   sampling_frac: float = None,
                   class_limit: int = None) -> Tuple[Sequence]:
    """

    :param path: path to data tree; should contain train and validation
    :param batch_size: size of batches used in training
    :param target_size: image size
    :param sample_frac: per class sampling
    :return: tuple of training data gen and validation generator
    """
    # pick the type of class to use
    if generator_type == 'image':
        training_data_cls = ImageGenerator
        validation_data_cls = ImageGenerator
    elif generator_type == 'text':
        training_data_cls = MetaDataGenerator
        validation_data_cls = MetaDataGenerator
    elif generator_type == 'multi':
        training_data_cls = ImageMetaDataGenerator
        validation_data_cls = ImageMetaDataGenerator
    else:
        raise ValueError('generator type not understood must be one of \
            [image, texxt, multi]')

    # build generators
    training_data = training_data_cls(
            primary_path + 'train/',
            batch_size,
            img_size,
            scaling=(1. / 255),
            sample_frac=sampling_frac,
            class_limit=class_limit
            )

    validation_data = validation_data_cls(
            primary_path + 'validation/',
            batch_size, 
            img_size,
            scaling=(1. / 255),
            sample_frac=sampling_frac,
            class_limit=class_limit
            )

    return training_data, validation_data
