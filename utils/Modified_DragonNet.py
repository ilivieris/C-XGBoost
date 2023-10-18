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


class EpsilonLayer(Layer):

    def __init__(self):
        super(EpsilonLayer, self).__init__()

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.epsilon = self.add_weight(name        = 'epsilon',
                                       shape       = [1, 1],
                                       initializer = 'RandomNormal',
                                       #  initializer='ones',
                                       trainable   = True)
        super(EpsilonLayer, self).build(input_shape)  # Be sure to call this at the end

    def call(self, inputs, **kwargs):
        # import ipdb; ipdb.set_trace()
        return self.epsilon * tf.ones_like(inputs)[:, 0:1]




def make_modified_dragonnet(input_dim, reg_l2, nOutcomesInfo = 2):
    """
    Neural net predictive model. The dragon has three heads.
    :param input_dim:
    :param reg:
    :return:
    """
    seed = 42
    tf.random.set_seed(seed) # for Tensoflow >= 2.0
    # tf.random.set_random_seed(seed)
    
    t_l1 = 0.
    t_l2 = reg_l2
    inputs = Input(shape=(input_dim,), name='input')
    
    input_y0 = Input(shape=(nOutcomesInfo // 2,), name='input_y0')
    input_y1 = Input(shape=(nOutcomesInfo // 2,), name='input_y1')
    
    
    # representation
    x = Dense(units=200, activation='elu', kernel_initializer='RandomNormal')(inputs)
    x = Dense(units=200, activation='elu', kernel_initializer='RandomNormal')(x)
    x = Dense(units=200, activation='elu', kernel_initializer='RandomNormal')(x)


    t_predictions = Dense(units=1, activation='sigmoid')(x)

    # Concatenate x with y
    Info_y0 = Concatenate(1)([x, input_y0])
    Info_y1 = Concatenate(1)([x, input_y1])
    
    # HYPOTHESIS
    y0_hidden = Dense(units=100, activation='elu', kernel_regularizer=regularizers.l2(reg_l2))(Info_y0)
    y1_hidden = Dense(units=100, activation='elu', kernel_regularizer=regularizers.l2(reg_l2))(Info_y1)

    # second layer
    y0_hidden = Dense(units=100, activation='elu', kernel_regularizer=regularizers.l2(reg_l2))(y0_hidden)
    y1_hidden = Dense(units=100, activation='elu', kernel_regularizer=regularizers.l2(reg_l2))(y1_hidden)

    # third
    y0_predictions = Dense(units=1, activation=None, kernel_regularizer=regularizers.l2(reg_l2), name='y0_predictions')(
        y0_hidden)
    y1_predictions = Dense(units=1, activation=None, kernel_regularizer=regularizers.l2(reg_l2), name='y1_predictions')(
        y1_hidden)

    
    
    dl       = EpsilonLayer()
    epsilons = dl(t_predictions, name='epsilon')
    # logging.info(epsilons)
    concat_pred = Concatenate(1)([y0_predictions, y1_predictions, t_predictions, epsilons])
    model       = Model(inputs=[inputs, input_y0, input_y1], outputs=concat_pred)

    return model