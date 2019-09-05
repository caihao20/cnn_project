#coding=GB18030

from math import ceil
import numpy as np
import time
from keras.applications import Xception
from keras.applications.inception_v3 import InceptionV3
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.layers import Dense, Dropout
from keras.models import Model
from keras.optimizers import *
from utils_dist import SGDRScheduler, CustomModelCheckpoint, SequenceData
from keras.callbacks import TensorBoard
from keras.utils import training_utils
from keras.utils import multi_gpu_model
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf

import os
#os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1'

np.random.seed(1024)

NBR_MODELS = 500

FINE_TUNE = False
warmup_epochs = 1
base_lr = 0.0125
momentum = 0.9
LEARNING_RATE = 0.0005
NBR_EPOCHS = 1500
BATCH_SIZE = 32
IMG_WIDTH = 299
IMG_HEIGHT = 299
monitor_index = 'val_acc'
USE_CLASS_WEIGHTS = False
RANDOM_SCALE = True
encoding = "gbk"
resume_from_epoch = 0

train_path = '/data/work_image/train_file/'
val_path = '/data/work_image/val_file/'
train_file_list = './data/500/train_file_796.txt'
val_file_list = './data/500/val_file_796.txt'
new_classes = ''
best_model_path = "./data/500/"
best_model_file = ""
last_mt_time = 0
##get the last modify h5 file
for maindir, subdirs, file_name_list in os.walk(best_model_path):
    for file in file_name_list:
        if file.find('.h5') == -1:
            continue

        mt_time = os.path.getmtime(os.path.join(maindir, file))
        if last_mt_time < mt_time:
            last_mt_time = mt_time
            best_model_file = os.path.join(maindir, file)
            FINE_TUNE = True


			
if __name__ == "__main__":
    # Pin a server GPU to be used by this process
    
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.9
    K.set_session(tf.Session(config=config))
    
    # ['/job:localhost/replica:0/task:0/device:CPU:0', '/job:localhost/replica:0/task:0/device:GPU:0', '/job:localhost/replica:0/task:0/device:GPU:1']
    nbr_gpus = len(training_utils._get_available_devices()) - 1

    verbose = 1 
    resume_from_epoch = 0 

    train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        width_shift_range=0.4,
        height_shift_range=0.4,
        rotation_range=90,
        zoom_range=0.7,
        horizontal_flip=True,
        vertical_flip=True)
    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        train_path,
        target_size=(299,299),
        batch_size=BATCH_SIZE,
        class_mode='categorical')

    val_generator = test_datagen.flow_from_directory(
        val_path,
        target_size=(299,299),
        batch_size=BATCH_SIZE,
        class_mode='categorical')        
    

    print('Loading Xception Weights ...')
    with tf.device('/cpu:0'):
        xception_path = 'xception_model/xception_weights_tf_dim_ordering_tf_kernels_notop.h5'
        inception = Xception(include_top=False, weights=xception_path,
                                input_tensor=None, input_shape=(IMG_WIDTH, IMG_HEIGHT, 3), pooling='avg')
        output = inception.get_layer(index=-1).output
        output = Dropout(0.5)(output)
        output = Dense(NBR_MODELS, activation='softmax', name='predictions')(output)
        for layer in inception.layers:
           layer.trainable = False
        model = Model(outputs=output, inputs=inception.input)

    if FINE_TUNE:
        print('Loading Xception Weights in file %s' % best_model_file)
        #model = multi_gpu_model(model, gpus=2)
        model.load_weights(best_model_file)

        if new_classes:
            with open(new_classes) as f:
                NBR_MODELS = len(f.readlines())

            f.close()
            print('use fine tune.....')
            output = model.get_layer(index=-2).output
            output = Dense(NBR_MODELS, activation='softmax', name='predictions')(output)
            model = Model(outputs=output, inputs=inception.input)

    print('Training model begins...')

    optimizer = SGD(lr=LEARNING_RATE , momentum=0.9, decay=0.0, nesterov=True)
    # optimizer = Adam(lr=LEARNING_RATE, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
    # optimizer = Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004)


    #model = multi_gpu_model(model, gpus=2) #若使用多GPU训练则去掉注释
    model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
  
    out_model_name = best_model_path + 'Xception_bestmodel_' + time.strftime('%y%m%d', time.localtime()) + "_500.h5"
    best_model = CustomModelCheckpoint(model, out_model_name, monitor_index=monitor_index)
    reduce_lr = ReduceLROnPlateau(monitor=monitor_index, factor=0.5, patience=10, verbose=1, min_lr=0)
    early_stop = EarlyStopping(monitor=monitor_index, patience=5, verbose=1, min_delta=0.001)

    try:
        train_data_lines = open(train_file_list, 'r', encoding=encoding).readlines()
    except UnicodeDecodeError:
        train_data_lines = open(train_file_list, 'r', encoding='UTF-8').readlines()
    node_nbr_train = len(train_data_lines)
    print('# Train Images: {}.'.format(node_nbr_train))
    steps_per_epoch = int(ceil(node_nbr_train / BATCH_SIZE))

    try:
        val_data_lines = open(val_file_list, 'r', encoding=encoding).readlines()
    except UnicodeDecodeError:
        val_data_lines = open(val_file_list, 'r', encoding='UTF-8').readlines()
    node_nbr_val = len(val_data_lines)
    print('# Val Images: {}.'.format(node_nbr_val))
    validation_steps = int(ceil(node_nbr_val / BATCH_SIZE))

    gpu_device_name = tf.test.gpu_device_name()
    print('gpu_device_name:',gpu_device_name)

    callbacks = list()
    callbacks.append(reduce_lr)
    callbacks.append(best_model)
    callbacks.append(early_stop)
    callbacks.append(TensorBoard(log_dir='./tb_log'))

    model.fit_generator(train_generator,
                        steps_per_epoch=steps_per_epoch, 
                        epochs=NBR_EPOCHS, verbose=verbose,
                        validation_data=val_generator,
                        validation_steps=validation_steps,
                        callbacks=callbacks,
                        shuffle=True,
                        max_queue_size=128, workers=4, use_multiprocessing=True)

