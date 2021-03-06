""" image processing utilities """
from pathlib2 import Path
import os
import numpy as np

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input

import IPython.display as display
from PIL import Image
import matplotlib.pyplot as plt

#############################################################################
######################## SET MEMORY GROWTH = TRUE ###########################
""" run this if you get ResourceExhaustedError """
physical_devices = tf.config.list_physical_devices('GPU')
try:
  tf.config.experimental.set_memory_growth(physical_devices[0], True)
  assert tf.config.experimental.get_memory_growth(physical_devices[0])
except:
  # Invalid device or cannot modify virtual devices once initialized.
  pass

#############################################################################
############################ PRE REQUISITS ##################################
#############################################################################

train_dir = Path('d:\\data\\train')
val_dir = Path('d:\\data\\val')
image_count = len(list(train_dir.glob('*/*.jpg'))) #5162596
val_image_count = len(list(val_dir.glob('*/*.jpg'))) #294099 
CLASS_NAMES = np.array([item.name for item in train_dir.iterdir()])
BATCH_SIZE = 39
IMG_HEIGHT = 150 #keep images small to save time
IMG_WIDTH = 150
STEPS_PER_EPOCH = np.ceil(image_count / BATCH_SIZE)

#############################################################################
#############################################################################
############################### TF DATA GENERATORS ##########################
#############################################################################
#############################################################################

def make_tfds(train_dir, val_dir):
    """ processes the training and validation sets """
    """ RUN DEPENDENT FUNCTIONS FIRST """
    """ dependencies: """
    """ process_path, prepare_for_training """
    #training image pathlist
    list_ds = tf.data.Dataset.list_files(str(train_dir / '*/*'))
    #validation image pathlist
    list_val_ds = tf.data.Dataset.list_files(str(val_dir / '*/*'))

    # setting up for parallel processing
    AUTOTUNE = tf.data.experimental.AUTOTUNE

    # before running, define process_path, 
    labeled_ds = list_ds.map(process_path, num_parallel_calls=AUTOTUNE)
    labeled_val_ds = list_val_ds.map(process_path, num_parallel_calls = AUTOTUNE)

    #before running, define prepare_for_training
    train_ds = prepare_for_training(labeled_ds) # cache = './img_store.tfcache'
    val_ds = prepare_for_training(labeled_val_ds)

    return train_ds, val_ds

def process_path(file_path):
  label = get_label(file_path)
  # load the raw data from the file as a string
  img = tf.io.read_file(file_path)
  img = decode_img(img)
  return img, label

def prepare_for_training(ds, cache=True, shuffle_buffer_size=1000):
  # use `.cache(filename)` to cache preprocessing work for datasets that don't
  # fit in memory.
  # just replace `True` with a filename, such as './img_cache.tfcache'
  if cache:
    if isinstance(cache, str):
      ds = ds.cache(cache)
    else:
      ds = ds.cache()

  ds = ds.shuffle(buffer_size=shuffle_buffer_size)

  # Repeat forever
  ds = ds.repeat()

  ds = ds.batch(BATCH_SIZE)

  # `prefetch` lets the dataset fetch batches in the background while the model
  # is training.
  ds = ds.prefetch(buffer_size=AUTOTUNE)

  return ds


##############################################################################
############################# MISC TOOLS #####################################
##############################################################################

def show_batch(image_batch, label_batch):
  plt.figure(figsize=(10,10))
  for n in range(25):
      ax = plt.subplot(5,5,n+1)
      plt.imshow(image_batch[n])
      plt.title(CLASS_NAMES[label_batch[n]==1][0].title())
      plt.axis('off')

def get_label(file_path):
  # convert the path to a list of path components
  parts = tf.strings.split(file_path, os.path.sep)
  # The second to last is the class-directory
  return parts[-2] == CLASS_NAMES

def decode_img(img):
  # convert the compressed string to a 3D uint8 tensor
  img = tf.image.decode_jpeg(img, channels=3)
  # Use `convert_image_dtype` to convert to floats in the [0,1] range.
  img = tf.image.convert_image_dtype(img, tf.float32)
  # resize the image to the desired size.
  return tf.image.resize(img, [IMG_HEIGHT, IMG_WIDTH])









#################################################################################
#################################################################################
################################ CNN MODELS #####################################
#################################################################################
#################################################################################


#####################################################################################
############################ FUNCTIONAL INCEPTION ###################################
#####################################################################################

""" If the object oriented version is run, then the verbiage won't provide the output
shapes of any layers. It has to do with the use of .super """

#load cifar to play with
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0
from tensorflow.keras.utils import to_categorical
# make labels categories instead of integers
train_labels, test_labels = to_categorical(train_labels), to_categorical(test_labels)

#add something to make the generator spit out 20 at a time?
# don't use this, it's not working and it's really not necessary
train_ds = (item for item in zip(train_images, train_labels))
val_ds = (item for item in zip(test_images, test_labels))


def FunctionalInception(input_shape, batch_size, num_classes):
    inputs = Input(shape = input_shape, batch_size = batch_size)
    x = Conv2D(10, 
               kernel_size = (1,1), 
               padding = 'same',
               activation = 'relu')(inputs)
    x = Conv2D(10, 
               kernel_size = (3,3),
               padding = 'same',
               activation = 'relu')(x)
    x2 = Conv2D(10, 
               kernel_size = (1,1), 
               padding = 'same',
               activation = 'relu')(inputs)
    x2 = Conv2D(10,
               kernel_size = (5,5),
               padding = 'same',
               activation = 'relu')(x2)
    x3 = MaxPooling2D(pool_size = (4,4),
                     strides = (1,1),
                     padding = 'same')(inputs)
    x3 = Conv2D(10, 
               kernel_size = (1,1), 
               padding = 'same',
               activation = 'relu')(x3)
    block = tf.keras.layers.concatenate([x, x2, x3], axis =3)

    x = Flatten()(block)
    x = Dense(1200, activation= 'relu')(x) #1200
    x = Dense(600, activation = 'relu')(x) #600
    x = Dense(150, activation  = 'relu')(x) #150
    x = Dense(num_classes, activation = 'softmax')(x)
    
    model = Model(inputs, x, name = 'FunctionalInception')

    return model

# THIS TRAINS ON CIFAR100. MODIFY FOR OUR DATA.
ncept_layer = FunctionalInception((32,32,3), 5, 10)
history = ncept_layer.fit(train_images, train_labels, 
              validation_data = (test_images, test_labels),
              #steps_per_epoch= STEPS_PER_EPOCH, comment out if not using generator
              #validation_steps= VAL_STEPS_PER_EPOCH,
              epochs = 4
              )
#####################################################################################
####################################### LENET-5 #####################################
#####################################################################################
""" an implementation of LeNet as a toy model. It's small and trains faster. """
""" Not a final model """
class LeNet5(Model):
    def __init__(self, num_classes):
        super(LeNet5, self).__init__()
        self.conv1 = Conv2D(2, 
                            kernel_size = (5,5), 
                            padding = 'same',
                            activation = 'relu')
        self.conv2 = Conv2D(8, 
                            kernel_size = (5,5),
                            padding = 'same',
                            activation = 'relu')
        self.max_pool = MaxPooling2D(pool_size = (3,3))
        self.flatten = Flatten()
        self.dense1 = Dense(120, activation= 'relu')
        self.dense2 = Dense(84, activation = 'relu')
        self.dense3 = Dense(num_classes, activation  = 'softmax')
    def call(self, x):
        x = self.max_pool(self.conv1(x))
        x = self.max_pool(self.conv2(x))
        x = self.flatten(x)
        x = self.dense3(self.dense2(self.dense1(x)))
        return x

toy_model = LeNet5(5000)
toy_model.compile(optimizer='sgd', 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

#what is the input shape?
# (batch_size, height, width, rgb channels)
input_shape = (3, 150, 150, 3)
toy_model.build(input_shape)
toy_model.summary()
history = toy_model.fit(train_ds, 
              validation_data = val_ds,
              steps_per_epoch= STEPS_PER_EPOCH,
              validation_steps= np.ceil(294099/39),
              epochs = 4
              )

###########################################################################33
############################ SINGLE INCEPTION LAYER #########################
#############################################################################
""" HAS BUG """
class Inception(Model):
    def __init__(self, num_classes):
        super(Inception, self).__init__()
        self.conv1 = Conv2D(10, 
                            kernel_size = (1,1), 
                            padding = 'same',
                            activation = 'relu')
        self.conv2 = Conv2D(10, 
                            kernel_size = (3,3),
                            padding = 'same',
                            activation = 'relu')
        self.conv3 = Conv2D(10,
                            kernel_size = (5,5),
                            padding = 'same',
                            activation = 'relu')
        self.max_pool = MaxPooling2D(pool_size = (3,3),
                                     strides = (1,1),
                                     padding = 'same')
        self.flatten = Flatten()
        self.dense1 = Dense(1200, activation= 'relu')
        self.dense2 = Dense(600, activation = 'relu')
        self.dense3 = Dense(150, activation  = 'relu')
        self.dense4 = Dense(num_classes, activation = 'softmax')
    def call(self, x):
        layer_1 = self.conv2(self.conv1(x))
        layer_2 = self.conv3(self.conv1(x))
        layer_3 = self.conv1(self.max_pool(x))
        x = tf.keras.layers.concatenate([layer_1, layer_2, layer_3], axis =3)
        x = self.flatten(x)
        x = self.dense4(self.dense3(self.dense2(self.dense1(x))))
        return x

batch_size = 39
STEPS_PER_EPOCH = np.ceil(train_images.shape[0] / batch_size)
VAL_STEPS_PER_EPOCH = np.ceil(test_images.shape[0] / batch_size)
mception = Inception(10)
mception.compile(optimizer='sgd', 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

input_shape = (39, 150, 150, 3)
mception.build(input_shape)
mception.summary()
history = mception.fit(train_images, train_labels, 
              validation_data = (test_images, test_labels),
              steps_per_epoch= STEPS_PER_EPOCH, #comment out if not using generator
              validation_steps= VAL_STEPS_PER_EPOCH, #same
              epochs = 4
              )


##########################################################################
##########################################################################
###################### KERAS DATA GENERATORS #############################
##########################################################################

train_datagen = ImageDataGenerator(
    rescale = 1./255,
    rotation_range = 40,
    width_shift_range = .2,
    height_shift_range = .2,
    shear_range = .2,
    zoom_range = .2,
    horizontal_flip = True,
    fill_mode = 'nearest')

# don't augment the validation data!
test_datagen = ImageDataGenerator(
    rescale = 1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size = (150, 150), # setting this to be small for now
    batch_size = 39, # try 39 because val_dir has 294099 images 
    class_mode = 'categorical')


val_generator = test_datagen.flow_from_directory(
    val_dir,
    target_size = (150, 150),
    batch_size = 39,
    class_mode = 'categorical')    

def limited_batch(generator, num):
    """ semi-functional : model will train as is, but better version in the works"""
    """ limits the number of batches the generator sends to the model """
    x = 0
    # example print-out that can be removed when used
    for data_batch, labels_batch in train_generator:
        
        # print('data batch shape: ', data_batch.shape)
        # print('labels batch shape: ', labels_batch.shape)
    #for i in train_generator:
        x += 1
        if x > num:
            break
        return data_batch, labels_batch
