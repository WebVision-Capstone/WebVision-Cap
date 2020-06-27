"""Data generators
"""

import glob
import os
import json
from pathlib import Path
from typing import Tuple, List

from PIL import Image
import numpy as np
from tensorflow.keras.utils import Sequence

class BaseDataGenerator(Sequence):
    """Keras-like data generator for image and metadata
    """
    
    def __init__(self,
                 path: str,
                 batch_size: int,
                 target_size: Tuple[int, int],
                 scaling: float = (1. / 255),
                 sample_frac: float = None) -> None:
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
        classes = sorted(glob.glob(self.path + '*'))
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

    def _load_metadata(self, path: str) -> Tuple[str, str]:
        """Load metadata from the JSON files
        """
        with open(Path(path).with_suffix(''), 'r') as in_file:
            items = json.load(in_file)
            return items['description'], items['title']

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
                 sample_frac: float = None) -> None:
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
                         sample_frac)
    
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
            desc, titl = self._load_metadata(f)
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

        img_reshape_size = [len(f_batch)] + list(self.target_size) + [3]
        
        # images, descriptions, titles, labels
        return ([np.array(images).reshape(*img_reshape_size),
               np.array(descriptions),
               np.array(titles)],
               np.array(labels))

class MetaDataGenerator(BaseDataGenerator):
    """Keras-like data generator for metadata
    """
    
    def __init__(self,
                 path: str,
                 batch_size: int,
                 target_size: Tuple[int, int],
                 scaling: float = (1. / 255),
                 sample_frac: float = None) -> None:
        """Constructor
        
        :param path: path to data tree
        :param batch_size: size of batches used in training
        :param target_size: image size
        :param scaling: image scaling value
        :param sample_frac: per class sampling
        """
        super().__init__(path,
                         batch_size,
                         (300, 300),
                         1. / 255,
                         sample_frac)
    
    
    def __getitem__(self, index: int) -> Tuple[List[np.ndarray], np.ndarray]:
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
            desc, titl = self._load_metadata(f)
            descriptions.append(desc)
            titles.append(titl)
            # get the label
            labels.append(
                self.label_mapping[
                    f.split('/')[-2]
                ]
            )
        
        # images, descriptions, titles, labels
        return ([np.array(descriptions),
               np.array(titles)],
               np.array(labels))

class ImageGenerator(BaseDataGenerator):
    """Keras-like data generator for image and metadata
    """
    
    def __init__(self,
                 path: str,
                 batch_size: int,
                 target_size: Tuple[int, int],
                 scaling: float = (1. / 255),
                 sample_frac: float = None) -> None:
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
                         sample_frac)
    
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
        return (np.array(images).reshape(*img_reshape_size),
               np.array(labels))

def get_generators(generator_type: str,
                   primary_path: str,
                   batch_size: int,
                   img_size: Tuple[int],
                   sampling_frac: float = None) -> Tuple[Sequence]:
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
            sample_frac=sampling_frac
            )

    validation_data = validation_data_cls(
            primary_path + 'validation/',
            batch_size, 
            img_size,
            scaling=(1. / 255),
            sample_frac=sampling_frac
            )

    return training_data, validation_data
