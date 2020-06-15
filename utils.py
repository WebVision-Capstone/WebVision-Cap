import glob
import json
from collections import defaultdict
from pathlib import Path

from PIL import Image
import numpy as np
from tensorflow.keras.utils import Sequence


class ImageMetaDataGenerator(Sequence):
    """Keras-like data generator for image and metadata
    """
    
    def __init__(self,
                 path,
                 batch_size,
                 target_size,
                 scaling = (1. / 255),
                 sample_frac = None) -> None:
        """Constructor
        
        :param path: path to data tree
        :param batch_size: size of batches used in training
        :param target_size: image size
        :param scaling: image scaling value
        :param sample_frac: per class sampling
        """
        self.path = path
        self.batch_size = batch_size
        self.target_size = target_size
        self.scaling = scaling
        self.sample_frac = sample_frac
        
        self.files = list()
        self.json_to_image_mapping = dict()
        self.label_mapping = dict()
        self.classes = list()
        

        # create a label maker
        classes = set(glob.glob(self.path + '*'))
        for i, label in enumerate(classes):
            label = label.split('/')[-1]
            self.classes.append(label)
            self.label_mapping[label] = i
        
        # get the file pathes
        # sample on a per class basis
        for class_path in classes:
            class_files = glob.glob(class_path + '/*.*')
            if self.sample_frac is not None:
                class_files = np.random.choice(class_files,
                                               np.floor(len(class_files) * self.sample_frac).astype(int),
                                               replace = False)
            self.files += list(class_files)

        print(f"Found {len(self.files)} instances belonging to {len(self.classes)} classes")

    def __len__(self):
        """Denotes the number of batches per epoch
        """
        return int(np.floor(len(self.files) / self.batch_size))
    
    def __getitem__(self, index):
        """Generate one batch of data
        """
        
        images, descriptions, titles = list(), list(), list()
        labels = list()

        f_batch = self.files[index * self.batch_size:(index + 1) * self.batch_size]
        for f in f_batch:
            # get the text data
            with open(Path(f).with_suffix(''), 'r') as in_file:
                items = json.load(in_file)
                descriptions.append(items['description'])
                titles.append(items['title'])
            # get the image
            img = Image.open(f)
            img = img.resize(self.target_size).convert('RGB')
            img = np.array(img) * self.scaling
            images.append(img)
            # get the label
            labels.append(
                self.label_mapping[
                    f.split('/')[-2]
                ]
            )

        img_reshape_size = [len(f_batch)] + list(self.target_size) + [3]

        return ([np.array(images).reshape(*img_reshape_size),
               np.array(descriptions),
               np.array(titles)],
               np.array(labels))
    
    def on_epoch_end(self):
        """Updates indexes after each epoch
        """
        np.random.shuffle(self.files)
