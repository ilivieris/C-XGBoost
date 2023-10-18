import warnings
warnings.filterwarnings("ignore")
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# Tensorflow library
#
import tensorflow               as tf
import tensorflow.keras.backend as K
from tensorflow.keras.metrics   import binary_accuracy
from tensorflow.keras.layers    import Input
from tensorflow.keras.layers    import Dense
from tensorflow.keras.layers    import Concatenate
from tensorflow.keras.layers    import BatchNormalization
from tensorflow.keras.layers    import Dropout
from tensorflow.keras.layers    import Layer
from tensorflow.keras.models    import Model
from tensorflow.keras           import regularizers

def make_ned(input_dim, reg_l2=0.01):
    """
    Neural net predictive model. The dragon has three heads.
    :param input_dim:
    :param reg:
    :return:
    """
    seed = 42
    tf.random.set_seed(seed)
    
    inputs = Input(shape=(input_dim,), name='input')

    # representation
    x = Dense(units=200, activation='elu', kernel_initializer='RandomNormal', name='ned_hidden1')(inputs)
    x = Dense(units=200, activation='elu', kernel_initializer='RandomNormal', name='ned_hidden2')(x)
    x = Dense(units=200, activation='elu', kernel_initializer='RandomNormal', name='ned_hidden3')(x)

    t_predictions = Dense(units=1, activation='sigmoid', name='ned_t_activation')(x)
    y_predictions = Dense(units=1, activation=None, name='ned_y_prediction')(x)

    concat_pred = Concatenate(1)([y_predictions, t_predictions])

    model = Model(inputs=inputs, outputs=concat_pred)
    return model


def post_cut(nednet, input_dim, reg_l2=0.01):
    
    seed = 42
    tf.random.set_seed(seed)
    
    for layer in nednet.layers:
        layer.trainable = False
    nednet.layers.pop()
    nednet.layers.pop()
    nednet.layers.pop()

    frozen = nednet

    x = frozen.layers[-1].output
 
#     frozen.layers[-1].outbound_nodes = []
    input = frozen.input

    y0_hidden = Dense(units=100, activation='elu', kernel_regularizer=regularizers.l2(reg_l2), name='post_cut_y0_1')(x)
    y1_hidden = Dense(units=100, activation='elu', kernel_regularizer=regularizers.l2(reg_l2), name='post_cut_y1_1')(x)

    # second layer
    y0_hidden = Dense(units=100, activation='elu', kernel_regularizer=regularizers.l2(reg_l2), name='post_cut_y0_2')(
        y0_hidden)
    y1_hidden = Dense(units=100, activation='elu', kernel_regularizer=regularizers.l2(reg_l2), name='post_cut_y1_2')(
        y1_hidden)

    # third
    y0_predictions = Dense(units=1, activation=None, kernel_regularizer=regularizers.l2(reg_l2), name='y0_predictions')(
        y0_hidden)
    y1_predictions = Dense(units=1, activation=None, kernel_regularizer=regularizers.l2(reg_l2), name='y1_predictions')(
        y1_hidden)

    concat_pred = Concatenate(1)([y0_predictions, y1_predictions])

    model = Model(inputs=input, outputs=concat_pred)
    
    return model

