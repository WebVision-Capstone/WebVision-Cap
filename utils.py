"""General utilities
"""

from typing import List

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
        ax[0].plot(history.history[item], label = item);
    ax[0].set_xlabel('Epochs');
    ax[0].set_ylabel('Accuracy');
    ax[0].legend();
    
    for item in loss_items:
        ax[1].plot(history.history[item], label = item);
    ax[1].set_xlabel('Epochs');
    ax[1].set_ylabel('Loss');
    ax[1].legend();
