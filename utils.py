import glob
import json

from PIL import Image
import numpy as np
from tensorflow.keras.utils import Sequence


class batch_generator(Sequence):
    
    def __init__(self,
                 path,
                 batch_size,
                 target_size,
                 scaling = (1. / 255)) -> None:
        """Constructor
        
        :param path: path to data tree
        :param batch_size: size of batches used in training
        :param target_size: image size
        :param scaling: image scaling value
        """
        self.path = path
        self.batch_size = batch_size
        self.batches = None
        self.files = None
        self.json_to_image_mapping = dict()
        self.label_mapping = dict()
        self.classes = None
        self.target_size = target_size
        self.scaling = scaling

        # get the file pathes
        self.files = glob.glob(self.path + '*/*.json', recursive=True)
        # map the json file to the image file
        for file in self.files:
            # get the json and image file
            tmp_files = glob.glob(file.replace('.json', '*'))
            for tf in tmp_files:
                if tf[-4:] != 'json':
                    # and make mapping
                    self.json_to_image_mapping[file] = tf
                    break

        # create a label maker
        self.classes = set(glob.glob(self.path + '*'))
        for i, label in enumerate(self.classes):
            label = label.split('/')[-1]
            self.label_mapping[label] = i

        # get a set of indicies
        idx = np.arange(len(self.files))
        # shuffle the input files
        np.random.shuffle(self.files)
        # get batch indicies
        self.batches = np.array_split(idx, len(self.files) // self.batch_size)

        print(f"Found {len(self.files)} instances belonging to {len(self.classes)} classes")

    def __len__(self):
        """Denotes the number of batches per epoch
        """
        return int(np.floor(len(self.files) / self.batch_size))
    
    def __getitem__(self, index):
        """Generate one batch of data
        """
        for batch in self.batches:
            images, descriptions, titles = list(), list(), list()
            labels = list()
            
            f_batch = self.files[index * self.batch_size:(index + 1) * self.batch_size]
            for f in f_batch:
                # get the text data
                with open(f, 'r') as in_file:
                    items = json.load(in_file)
                    descriptions.append(items['description'])
                    titles.append(items['title'])
                # get the image
                img = Image.open(self.json_to_image_mapping[f])
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
