import tensorflow as tf


def causal_siamese_loss(margin:int=2, alpha:float=0.9):
    """ 
    Provides causal siamese loss whcih is composed by
    (i) "contrastive loss" an enclosing scope with variable 'margin'.
    (ii) "binary crossentropy loss"

    Parameters
    ----------
        margin: defines the baseline for distance for which pairs
                should be classified as dissimilar. - (default is 1).
        alpha: float 0<alpha<1, defines the weight between contrastive loss and binary crossentropy loss

    Returns
    -------
        Loss 
    """

    # Contrastive loss = mean( (1-true_value) * square(prediction) +
    #                         true_value * square( max(margin-prediction, 0) ))
    def loss(yt_true, yt_pred):
        """
        Calculates the contrastive loss

        Parameters
        ----------
            y_true: List of labels, each label is of type float32.
            y_pred: List of predictions of same length as of y_true,
                    each label is of type float32.

        Returns
        -------
            A tensor containing contrastive loss as floating point value.
        """ 
        y_true, t_true = yt_true[:,0], yt_true[:,1]
        y0_pred, y1_pred, t_pred, epsilons = yt_pred[:,0], yt_pred[:,1], yt_pred[:,2], yt_pred[:,3]
        t_pred   = (t_pred + 0.01) / 1.02

        # Contrastive Loss
        y_pred = (1.0 - t_true) * y0_pred + t_true * y1_pred 

        square_pred = tf.math.square(y_pred)
        margin_square = tf.math.square(tf.math.maximum(margin - y_pred, 0))
        contrastive_loss = tf.math.reduce_mean( (1.0 - y_true) * square_pred + (y_true) * margin_square )

        # Binary Crossentropy
        binary_crossentropy_loss = tf.keras.losses.BinaryCrossentropy()(t_true, t_pred)


        return alpha*contrastive_loss + (1-alpha)*binary_crossentropy_loss


        # # Control group
        # square_pred = tf.math.square(y0_pred)
        # margin_square = tf.math.square(tf.math.maximum(margin - y0_pred, 0))
        # control_group_loss = (1. - y_true) * square_pred + (y_true) * margin_square
        # # Treatment group
        # square_pred = tf.math.square(y_true - y1_pred)
        # margin_square = tf.math.square(tf.math.maximum(margin - (y_true - y1_pred), 0))
        # treatment_group_loss = (1. - y_true) * square_pred + (y_true) * margin_square

        # # Loss
        # return tf.math.reduce_mean( (1. - t_true)*control_group_loss + t_true*treatment_group_loss )

    return loss











# def tarreg_contrastive_loss(margin=2, alpha=0.9, ratio=1.):
#     """ # LIVIERIS
#     Provides 'contrastive loss' an enclosing scope with variable 'margin',
#     enforced with target regularization

#     Parameters
#     ----------
#         margin: Integer, defines the baseline for distance for which pairs
#                 should be classified as dissimilar. - (default is 1).

#     Returns
#     -------
#         'contrastive_loss' function with data ('margin') attached.
#     """

#     # Contrastive loss = mean( (1-true_value) * square(prediction) +
#     #                         true_value * square( max(margin-prediction, 0) ))
#     def causal_siamese_loss(yt_true, yt_pred):
#         """
#         Calculates the contrastive loss

#         Parameters
#         ----------
#             y_true: List of labels, each label is of type float32.
#             y_pred: List of predictions of same length as of y_true,
#                     each label is of type float32.

#         Returns
#         -------
#             A tensor containing contrastive loss as floating point value.
#         """ 
#         y_true, t_true = yt_true[:,0], yt_true[:,1]
#         y0_pred, y1_pred, t_pred, epsilons = yt_pred[:,0], yt_pred[:,1], yt_pred[:,2], yt_pred[:,3]
#         t_pred   = (t_pred + 0.01) / 1.02

#         # Contrastive Loss
#         y_pred = (1.0 - t_true) * y0_pred + t_true * y1_pred 

#         square_pred = tf.math.square(y_pred)
#         margin_square = tf.math.square(tf.math.maximum(margin - y_pred, 0))
#         contrastive_loss = tf.math.reduce_mean( (1.0 - y_true) * square_pred + (y_true) * margin_square )

#         # Binary Crossentropy
#         binary_crossentropy_loss = tf.keras.losses.BinaryCrossentropy()(t_true, t_pred)

#         # Targeted regularization
#         h = t_true / t_pred - (1 - t_true) / (1 - t_pred)

#         y_pert = y_pred + epsilons * h
#         targeted_regularization = tf.reduce_sum(tf.square(y_true - y_pert))

#         return alpha*contrastive_loss + (1-alpha)*binary_crossentropy_loss + ratio*targeted_regularization


#     return causal_siamese_loss