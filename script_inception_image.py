# train an image model with inception

import glob
import os
import pickle

import matplotlib.pyplot as plt
import tensorflow as tf

from generators import get_generators
from utils import (
    plot_accuracy_loss,
    getArgs
    )
from models import get_inception_v3

from const import DATA_PATH


if __name__ == "__main__":

    # unpack arguments
    args = getArgs()
    epochs = args.epochs
    save_path = args.save_path
    job_id = args.job_id
    batch_size = args.batch_size
    img_size = args.img_size
    workers = args.workers
    unfreeze_layer = args.unfreeze_layer
    path_to_data = args.path_to_data

    if path_to_data == '':
        path_to_data = DATA_PATH

    strategy = tf.distribute.MirroredStrategy()
    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

    training_data, validation_data = get_generators('image',
                                                    path_to_data,
                                                    batch_size,
                                                    (img_size, img_size)
                                                    )

    # save on each epoch
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=save_path + 'checkpoints/cp-' 
            + job_id 
            + '{epoch:04d}.ckpt',
        save_weights_only=True,
        )                  


    with strategy.scope():
        inception_model = get_inception_v3((img_size, img_size),
                                           unfreeze_layer)

        inception_model.compile(optimizer='adam', 
                                loss='sparse_categorical_crossentropy', 
                                metrics=['accuracy', 'sparse_top_k_categorical_accuracy'])

    # fit the model
    image_hist = inception_model.fit(training_data,
                                    epochs = epochs,
                                    validation_data = validation_data,
                                    callbacks=[model_checkpoint_callback],
                                    workers = workers
                                    )

    with open(save_path + job_id + '-hist.pickle', 'wb') as f:
        pickle.dump(image_hist, f)
