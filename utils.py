"""General utilities
"""

from typing import List
import argparse

import matplotlib.pyplot as plt
import tensorflow as tf

def plot_accuracy_loss(history: tf.keras.callbacks.History,
                       acc_items: List[str],
                       loss_items: List[str]):
    """Plot accuracy metrics and loss metrics from training history
    
    Usage:
    plot_accuracy_loss(
                       history,
                       ['accuracy', 'val_accuracy'],
                       ['loss', 'val_loss'])
    
    """
    fig, ax = plt.subplots(ncols = 2, figsize = (15, 7))
    fig.suptitle('Accuracy and Loss')
    
    for item in acc_items:
        ax[0].plot(history.history[item], label = item)
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('Accuracy')
    ax[0].legend()
    
    for item in loss_items:
        ax[1].plot(history.history[item], label = item)
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('Loss')
    ax[1].legend()

def getArgs(argv=None):
    parser = argparse.ArgumentParser(
        description='Get config inputs')

    parser.add_argument('epochs', 
                        metavar='epochs', 
                        type=int, 
                        nargs='?', 
                        action='store',
                        help='number of training epochs'
                        )

    parser.add_argument('save_path', 
                        metavar='save_path', 
                        type=str, 
                        nargs='?', 
                        action='store',
                        help='path to save objects (weights, pickles)'
                        )

    parser.add_argument('job_id', 
                        metavar='job_id', 
                        type=str, 
                        nargs='?', 
                        action='store',
                        help='name or id of job'
                        )

    parser.add_argument('--batch_size',
                        metavar='batch_size',
                        type=int,
                        nargs='?',
                        default=64,
                        help='generator batch size'
                        )

    parser.add_argument('--img_size', 
                        metavar='img_size', 
                        type=int, 
                        nargs='?',
                        default=300,
                        help='size of the input image (one side of square)'
                        )

    parser.add_argument('--workers', 
                        metavar='workers', 
                        type=int, 
                        nargs='?',
                        default=1,
                        help='number of threads to load data'
                        )

    parser.add_argument('--unfreeze_layer', 
                        metavar='workers', 
                        type=str, 
                        nargs='?',
                        default='',
                        help='unfreeze layers after this one (ConvNet tower)'
                        )

    parser.add_argument('--path_to_data', 
                        metavar='pickle', 
                        type=str, 
                        nargs='?',
                        default='',
                        help='pickle the model history'
                        )

    return parser.parse_args()