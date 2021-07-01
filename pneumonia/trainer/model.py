import os
import shutil
import datetime

import hypertune

import numpy as np
import pandas as pd
import pathlib

#import cv2

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras.layers import Dense, Flatten, Softmax

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D,GlobalAveragePooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

import warnings
warnings.filterwarnings('ignore')

import logging
logger = tf.get_logger()
logger.setLevel(logging.ERROR)


def load_data(train_path, val_path, batch_size):
    
    CLASS_LABELS = ['NORMAL', 'PNEUMONIA'] 

    def process_path(nb_class):
    
        def f(file_path):
            
            label = 0 if tf.strings.split(file_path, os.path.sep)[-2]=='NORMAL' else 1
            
            image = tf.io.read_file(file_path)    
            image = tf.image.decode_jpeg(image, channels=3)
            image = tf.image.convert_image_dtype(image, tf.float32)
         
            image = tf.image.resize(image, [127, 127], method='area')
            return image, label
    
        return f

    def reader_image(path_file, batch_size, nb_class):

        list_ds = tf.data.Dataset.list_files(path_file)
        labeled_ds = list_ds.map(process_path(nb_class))
    
        return labeled_ds.shuffle(100).batch(batch_size).prefetch(1)
    
    train_ds = reader_image(train_path, batch_size, 2)
    val_ds = reader_image(val_path, batch_size, 2)

   # train_ds = reader_image('gs://chest-xray-us-central/chest_xray/train/*/*.jpeg', 16, 2)
   # val_ds = reader_image('gs://chest-xray-us-central/chest_xray/test/*/*.jpeg', 16, 2)
    print(type(train_ds))


    for image, label in train_ds.take(1):
        df = pd.DataFrame(image[0, :, :, 0].numpy())
    
    print(f'Outoupt : \n image shape: {df.shape}')
    
    return train_ds, val_ds

def train_and_evaluate(args):
    from tensorflow.keras.applications.densenet import DenseNet169
    from tensorflow.keras.applications.densenet import preprocess_input as densenet_preprocess

    base = DenseNet169(weights = 'imagenet', include_top = False, input_shape = (127, 127, 3))
    tf.keras.backend.clear_session()

    for layer in base.layers:
        layer.trainable =  False 

    densenet_model = Sequential()
    densenet_model.add(base)
    densenet_model.add(GlobalAveragePooling2D())
    densenet_model.add(BatchNormalization())
    densenet_model.add(Dense(256, activation='relu'))
    densenet_model.add(Dropout(0.5))
    densenet_model.add(BatchNormalization())
    densenet_model.add(Dense(128, activation='relu'))
    densenet_model.add(Dropout(0.5))
    densenet_model.add(Dense(1, activation='sigmoid'))

    densenet_model.summary()
    
    eval_steps = args["eval_steps"]
    
    optm = Adam(lr=0.0001)
    densenet_model.compile(loss='binary_crossentropy', optimizer=optm, 
                  metrics=['accuracy'])

    checkpoint_path = os.path.join(args["output_dir"], "checkpoints/pneumonia")
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path, verbose=1, save_weights_only=True)
    
    train_ds, val_ds = load_data(args["train_data_path"], args["eval_data_path"], args["batch_size"])
  
    dense_history = densenet_model.fit(
              train_ds,
              validation_data=val_ds,
              epochs=args["num_epochs"])
    print("cheking the model run")
    
    EXPORT_PATH = os.path.join(
        args["output_dir"], datetime.datetime.now().strftime("%Y%m%d%H%M%S"))
    tf.saved_model.save(
        obj=densenet_model, export_dir=EXPORT_PATH)
    
    print("Exported trained model to {}".format(EXPORT_PATH))
    

    hp_metric = dense_history.history['val_accuracy'][eval_steps-1]
    
    hpt = hypertune.HyperTune()
    hpt.report_hyperparameter_tuning_metric(
        hyperparameter_metric_tag='accuracy',
        metric_value=hp_metric,
        global_step=eval_steps
    )
    return dense_history
    
