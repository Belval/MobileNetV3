import tensorflow as tf
from tensorflow.keras import backend as K


def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection) / (K.sum(y_true_f) + K.sum(y_pred_f))

def dice_coef_multilabel_builder(num_labels):
    def dice_coef_multilabel(y_true, y_pred):
        dice=0
        for index in range(num_labels):
            dice += (1 - dice_coef(y_true[:,:,:,index], y_pred[:,:,:,index])) / num_labels
        return dice

    return dice_coef_multilabel
