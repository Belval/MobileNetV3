import tensorflow as tf
from tensorflow.keras import backend as K


def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2.0 * intersection) / (K.sum(y_true_f) + K.sum(y_pred_f))


def dice_coef_multilabel_builder(num_labels):
    def dice_coef_multilabel(y_true, y_pred):
        dice = 0
        for index in range(num_labels):
            dice += (
                1 - dice_coef(y_true[:, :, :, index], y_pred[:, :, :, index])
            ) / num_labels
        return dice

    return dice_coef_multilabel


def jaccard_distance(y_true, y_pred, smooth=100):
    if K.sum(y_pred) == 0:
        # When training two heads, we might not have a mask
        return K.sum(y_pred)
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    sum_ = K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return (1 - jac) * smooth


def weighted_categorical_crossentropy(weights):
    """
    A weighted version of keras.objectives.categorical_crossentropy
    
    Variables:
        weights: numpy array of shape (C,) where C is the number of classes
    
    Usage:
        weights = np.array([0.5,2,10]) # Class one at 0.5, class 2 twice the normal weights, class 3 10x.
        loss = weighted_categorical_crossentropy(weights)
        model.compile(loss=loss,optimizer='adam')
    """

    weights = K.variable(weights)

    def loss(y_true, y_pred):
        if K.sum(y_pred) == 0:
            return 0
        # scale predictions so that the class probas of each sample sum to 1
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        # clip to prevent NaN's and Inf's
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        # calc
        loss = y_true * K.log(y_pred) * weights
        loss = -K.sum(loss, -1)
        return loss

    return loss
