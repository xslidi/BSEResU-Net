###################################################
#
#   Script to:
#   - Load the images and extract the patches
#   - Define the neural network
#   - define the training
#
##################################################
# -------------------------- set gpu using tf ---------------------------
from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)
# -----------------------------------------------------------------------


import numpy as np
import configparser

from tensorflow.keras import Model
from tensorflow.keras.layers import Input, concatenate, Conv2D, MaxPooling2D, UpSampling2D, Reshape, Dropout, Conv2DTranspose, Multiply
from tensorflow.keras.optimizers import Adam, Nadam
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, CSVLogger, TensorBoard, EarlyStopping
from tensorflow.keras import backend as K
from tensorflow.keras.utils import plot_model
from tensorflow.keras.optimizers import SGD, schedules
from tensorflow.keras.layers import Dense, Lambda, Activation, Permute, Flatten
from tensorflow.keras import layers
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import BatchNormalization, LayerNormalization  
from tensorflow.keras.losses import categorical_crossentropy
from DropBlock import DropBlock2D, Mish



import sys
sys.path.insert(0, './lib/')
from help_functions import *

#function to obtain data for training/testing (validation)
from extract_patches import get_data_training


def dice_coef(y_true, y_pred, smooth=1):
    """
    Dice = (2*|X & Y|)/ (|X|+ |Y|)
         =  2*sum(|A*B|)/(sum(A^2)+sum(B^2))
    ref: https://arxiv.org/pdf/1606.04797v1.pdf
    """
    y_true_f = K.flatten(y_true[:,:,1])
    y_pred_f = K.flatten(y_pred[:,:,1])
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    # return (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def combo_loss(y_true, y_pred):
    dice_loss = K.mean(1-dice_coef(y_true, y_pred))
    loss = 0.25*dice_loss  + 0.75*categorical_crossentropy(y_true, y_pred)
    return loss

def hw_flatten(x):
    return tf.reshape(x, [K.shape(x)[0],-1,x.shape[-1]])


def ChannelSE(reduction=4):
    """
    Squeeze and Excitation block, reimplementation inspired by
        https://github.com/Cadene/pretrained-models.pytorch/blob/master/pretrainedmodels/models/senet.py
    Args:
        reduction: channels squeeze factor
    """

    def layer(input_tensor):
        # get number of channels/filters
        channels = K.int_shape(input_tensor)[3]

        x = input_tensor

        # squeeze and excitation block in PyTorch style with
        # custom global average pooling where keepdims=True
        x = Lambda(lambda a: K.mean(a, axis=[1, 2], keepdims=True))(x)
        x = Conv2D(channels // reduction, (1, 1), kernel_initializer='he_uniform', data_format='channels_last')(x)
        
        x = Activation('relu')(x)
        x = Conv2D(channels, (1, 1), kernel_initializer='he_uniform', data_format='channels_last')(x)
        x = Activation('sigmoid')(x)

        # apply attention
        x = Multiply()([input_tensor, x])

        return x

    return layer


# Define buildingbolck unit like ResNet
def residual_block(input_tensor, kernel_size, filters, stage, block, add_bn=True,
    dropout=False, pool=True, SE=True, reduction=4, GC=False, GP=False):
    """The identity block is the block that has no conv layer at shortcut.
    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
        add_bn: bool. True if you want to add BatchNormalization layer.
    # Returns
        Output tensor for the block.
    """
    filters1, filters2 = filters
    groups = 2
    radix = 2
    bn_axis = 3
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    if pool:
        stride = (1, 1)
        filtersin = filters1
        
    else:
        stride = (2, 2)
        filtersin = filters1 // 2
    x = input_tensor
    if add_bn:
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)

    x = Conv2D(filters1, (kernel_size, kernel_size), strides=stride,
             padding='same', name=conv_name_base + '2a', data_format='channels_last')(x)

    x = Activation('relu')(x)

    if add_bn:
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)

    if dropout:
        x = DropBlock2D(7, 0.8)(x)

    x = Conv2D(filters2, (kernel_size, kernel_size), padding='same',
                name=conv_name_base + '2c', data_format='channels_last')(x)

    x = Activation('relu')(x)
    
    if SE:
        x = ChannelSE(reduction=reduction)(x)

    if x.shape != input_tensor.shape:
        shortcut = Conv2D(filters2, (1, 1), strides=stride,
                        padding='same', name=conv_name_base + '1', data_format='channels_last')(input_tensor)
    else:
        shortcut = input_tensor
    x = layers.add([x, shortcut])

    return x


# Define the resU-net without maxpooling which has 3 blocks
def bseresunet(patch_height,patch_width,n_ch):
    width = 1
    original_inputs = Input(shape=(patch_height,patch_width,n_ch))
    # K.image_data_format() == 'channels_last'
    conv1 = residual_block(original_inputs, 3, [32*width, 32*width], stage=1, block='a', dropout=True, GP=False)
    conv1 = residual_block(conv1, 3, [32*width, 32*width], stage=1, block='b', dropout=True)
    #conv1 = residual_block(conv1, 3, [32*width, 32*width], stage=1, block='c', dropout=True)
    #
    conv2 = residual_block(conv1, 3, [64*width, 64*width], stage=2, block='a', dropout=True, pool=False)
    conv2 = residual_block(conv2, 3, [64*width, 64*width], stage=2, block='b', dropout=True)
    #conv2 = residual_block(conv2, 3, [64*width, 64*width], stage=2, block='c', dropout=True)

    #
    conv3 = residual_block(conv2, 3, [128*width, 128*width], stage=3, block='a', dropout=True, pool=False)
    conv3 = residual_block(conv3, 3, [128*width, 128*width], stage=3, block='b', dropout=True)
    #conv3 = residual_block(conv3, 3, [128*width, 128*width], stage=3, block='c', dropout=True)
    #conv3 = residual_block(conv3, 3, [128*width, 128*width], stage=3, block='d', dropout=True)


    up1 = Conv2DTranspose(128*width, (3, 3), activation='relu', padding='same', strides=(2, 2), data_format='channels_last')(conv3)
    up1 = concatenate([conv2,up1],axis=3)
    conv4 = residual_block(up1, 3, [64*width, 64*width], stage=4, block='a', dropout=True)
    conv4 = residual_block(conv4, 3, [64*width, 64*width], stage=4, block='b', dropout=True)
    #conv4 = residual_block(conv4, 3, [64*width, 64*width], stage=4, block='c', dropout=True)

    #
    up2 = Conv2DTranspose(64*width, (3, 3), activation='relu', padding='same', strides=(2, 2), data_format='channels_last')(conv4)
    
    up2 = concatenate([conv1,up2], axis=3)
    conv5 = residual_block(up2, 3, [32*width, 32*width], stage=5, block='a', dropout=True)
    conv5 = residual_block(conv5, 3, [32*width, 32*width], stage=5, block='b', dropout=True)
    #conv5 = residual_block(conv5, 3, [32*width, 32*width], stage=5, block='c', dropout=True)
    #
    conv6 = Conv2D(2, (1, 1), activation='relu',padding='same', data_format='channels_last')(conv5)
    conv6 = Reshape((patch_height*patch_width,2))(conv6)
    ############
    conv7 = Activation('softmax')(conv6)

    model = Model(inputs=original_inputs, outputs=conv7)

    sgd = SGD(lr=0.02, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss=combo_loss, metrics=['accuracy'])

    return model


#========= Load settings from Config file
config = configparser.RawConfigParser()
config.read('configuration.txt')
#patch to the datasets
path_data = config.get('data paths', 'path_local')
#Experiment name
name_experiment = config.get('experiment name', 'name')
#training settings
N_epochs = int(config.get('training settings', 'N_epochs'))
batch_size = int(config.get('training settings', 'batch_size'))

#============ Load the data and divided in patches
patches_imgs_train, patches_masks_train = get_data_training(
    DRIVE_train_imgs_original = path_data + config.get('data paths', 'train_imgs_original'),
    DRIVE_train_groudTruth = path_data + config.get('data paths', 'train_groundTruth'),  #masks
    patch_height = int(config.get('data attributes', 'patch_height')),
    patch_width = int(config.get('data attributes', 'patch_width')),
    N_subimgs = int(config.get('training settings', 'N_subimgs')),
    inside_FOV = config.getboolean('training settings', 'inside_FOV') #select the patches only inside the FOV  (default == True)
)


#========= Save a sample of what you're feeding to the neural network ==========
N_sample = min(patches_imgs_train.shape[0],40)
visualize(group_images(patches_imgs_train[0:N_sample,:,:,:],5),'./'+name_experiment+'/'+"sample_input_imgs")#.show()
visualize(group_images(patches_masks_train[0:N_sample,:,:,:],5),'./'+name_experiment+'/'+"sample_input_masks")#.show()


#=========== Construct and save the model arcitecture =====
n_ch = patches_imgs_train.shape[3]
patch_height = patches_imgs_train.shape[1]
patch_width = patches_imgs_train.shape[2]
# print()
model = bseresunet(patch_height, patch_width,n_ch)  #the U-net model
print ("Check: final output of the network:")
print (model.output_shape)
# plot_model(model, to_file='./'+name_experiment+'/'+name_experiment + '_model.png')   #check how the model looks like
json_string = model.to_json()
open('./'+name_experiment+'/'+name_experiment +'_architecture.json', 'w').write(json_string)



#============  Training ==================================
#save at each epoch if the validation decreased
checkpointer = ModelCheckpoint(filepath='./'+name_experiment+'/'+name_experiment +'_best_weights.h5',
                    verbose=1, monitor='val_loss', mode='auto', save_best_only=True)


def step_decay(epoch):
    lrate = 0.02 #the initial learning rate (by default in keras)
    if epoch <= 30:
        return lrate
    elif epoch <= 50:
        return lrate*0.5
    else:
        return lrate*0.25

lrate_drop = LearningRateScheduler(step_decay, verbose=1)
early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=20, verbose=1, mode='auto', baseline=None, restore_best_weights=False)


csv_logger = CSVLogger('./test/training.csv')
visualize = TensorBoard(log_dir='./test/logs')
patches_masks_train = masks_Unet(patches_masks_train)  #reduce memory consumption
model.fit(patches_imgs_train, patches_masks_train, epochs=N_epochs,
          batch_size=batch_size, verbose=2, shuffle=True, validation_split=0.1,
          callbacks=[checkpointer, csv_logger, visualize, lrate_drop, early_stop])


#========== Save and test the last model ===================
model.save_weights('./'+name_experiment+'/'+name_experiment +'_last_weights.h5', overwrite=True)
#test the model
# score = model.evaluate(patches_imgs_test, masks_Unet(patches_masks_test), verbose=0)
# print('Test score:', score[0])
# print('Test accuracy:', score[1])
