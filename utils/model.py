import tensorflow               as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers    import Input
from tensorflow.keras.layers    import Dense
from tensorflow.keras.layers    import Concatenate
from tensorflow.keras.layers    import BatchNormalization
from tensorflow.keras.layers    import Layer, Lambda
from tensorflow.keras.models    import Model
from tensorflow.keras           import regularizers



def euclidean_distance(vects):
    """
    Find the Euclidean distance between two vectors.

    Parameters
    ----------
        vects: List containing two tensors of same length.

    Returns
    -------
        Tensor containing euclidean distance
        (as floating point value) between vectors.
    """

    x, y = vects
    sum_square = tf.math.reduce_sum(tf.math.square(x - y), axis=1, keepdims=True)
    
    return tf.math.sqrt(tf.math.maximum(sum_square, tf.keras.backend.epsilon()))

L1_distance = Lambda(lambda tensors: K.abs(tensors[0] - tensors[1]))


class EpsilonLayer(Layer):
    '''
        Create Epsilon layer
    '''
    def __init__(self):
        super(EpsilonLayer, self).__init__()

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.epsilon = self.add_weight(name='epsilon',
                                       shape=[1, 1],
                                       initializer='RandomNormal',
                                       #  initializer='ones',
                                       trainable=True)
        super(EpsilonLayer, self).build(input_shape)  # Be sure to call this at the end

    def call(self, inputs, **kwargs):
        # import ipdb; ipdb.set_trace()
        return self.epsilon * tf.ones_like(inputs)[:, 0:1]
    

# Backbone network
# ------------------------------------------------------------------------
def backbone_network(number_of_covariates:int=None)->Model:
    '''
        Create Siamese's backbone network

        Parameters
        ----------
        number_of_covariates: number of covariates

        Returns
        -------
        backbone model        
    '''
    
    inputs = Input(shape=(number_of_covariates,), name='input')

    x = Dense(units=200, activation='elu', kernel_initializer='RandomNormal')(inputs)
    x = Dense(units=200, activation='elu', kernel_initializer='RandomNormal')(x)
    x = Dense(units=200, activation='elu', kernel_initializer='RandomNormal')(x)
    
    return Model(inputs, x)




def Causal_Siamese_network(number_of_covariates:int=None, reg_l2:float=1e-2, distance:str="L2")->Model:
    '''
        Create Causal Siamese network


        Parameters
        ----------
        number_of_covariates: number of covariates
        reg_l2: regularization L2 parameter
        distance: selected distance {"L1", "L2"}

        Returns
        -------
        Prediction model
    '''

    # Inputs
    inputs0 = Input(shape=(number_of_covariates,), name='input0') # Anchor
    inputs1 = Input(shape=(number_of_covariates,), name='input1') # Similar/Non-similar instance

    # Input embeddings
    embedding_network = backbone_network(number_of_covariates=number_of_covariates)
    embedding_network1_embeddings0 = embedding_network(inputs0)
    embedding_network1_embeddings1 = embedding_network(inputs1)

    # Part 1: Propensity score g(\theta)
    t_predictions = Dense(1, activation="sigmoid", kernel_regularizer=regularizers.l2(1e-4), name="treatment_output")(embedding_network1_embeddings0)

    dl = EpsilonLayer()
    epsilons = dl(t_predictions, name='epsilon')


    # Part 2: Siamese network

    if distance == "L2":
        merge_layer = Lambda(euclidean_distance)([embedding_network1_embeddings0, embedding_network1_embeddings1])
    elif distance == "L1":
        merge_layer = L1_distance([embedding_network1_embeddings0, embedding_network1_embeddings1])
    else:
        raise Exception("Not known distance")
    
    normal_layer = tf.keras.layers.BatchNormalization()(merge_layer)
    similarity_predictions = Dense(2, activation="sigmoid", kernel_regularizer=regularizers.l2(reg_l2), name='similarity_output')(normal_layer)

    
    return Model(inputs=[inputs0, inputs1], 
                 outputs=Concatenate(axis=1, name="output")([similarity_predictions, t_predictions, epsilons]),
                 name="Causal_Siamese_network")