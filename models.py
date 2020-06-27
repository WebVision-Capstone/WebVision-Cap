"""Models
"""

from typing import Tuple, List

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.applications.inception_v3 import InceptionV3


def unfreeze_idx(layers: List[tf.keras.layers.Layer],
                 layer_name: str
                 ) -> int:
    """Find index of layer
    """
    last_trainable_layer = int()
    for i, layer in enumerate(layers):
        if layer.name == layer_name:
            last_trainable_layer = i
            break
    last_trainable_layer += 1
    return last_trainable_layer

def get_inception_v3(target_size: Tuple[int],
                     unfreeze_layer: str = ''
                     ) -> Model:
    """
    :param target_size: the image size
    :param unfeeze_layer: layer to start unfreezing the model (e.g. 'mixed5')
    :return: uncompiled keras model
    """

    pre_trained_model = InceptionV3(input_shape = target_size + tuple([3]), 
                                    include_top = False)

    if unfreeze_layer != '':
        # find the last module before opening for training
        for i, layer in enumerate(pre_trained_model.layers):
            if layer.name == unfreeze_layer:
                last_trainable_layer = i
                break
        last_trainable_layer += 1

        print("Unfreezing from " 
            + pre_trained_model.layers[last_trainable_layer-1].name)

        # lock the early layers
        for layer in pre_trained_model.layers[:last_trainable_layer]:
            layer.trainable = False

        # unlock the later layers
        for layer in pre_trained_model.layers[last_trainable_layer:]:
            layer.trainable = True
    else:
        for layer in pre_trained_model.layers:
            layer.trainable = False
        
    # use mixed10 as the last layer from inception3
    last_layer = pre_trained_model.get_layer('mixed10')
    last_output = last_layer.output

    # Flatten the output layer to 1 dimension
    x = tf.keras.layers.GlobalAveragePooling2D()(last_output)

    #x = Dense(2048, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.2)(x)

    output = tf.keras.layers.Dense(5000, activation='softmax', name='output')(x)

    return Model(inputs=[pre_trained_model.input], outputs=[output])
