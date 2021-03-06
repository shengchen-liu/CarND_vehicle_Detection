### Import libraries

import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import glob

import keras
from keras.models import Model
from keras.layers import Input, concatenate, Convolution2D, MaxPooling2D, UpSampling2D,Lambda
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler,CSVLogger, TensorBoard
from keras import backend as K
from scipy.ndimage.measurements import label
import time

### Make data frame in Pandas

import pandas as pd
from load_data import load_data, split_train_val
from utils import *
from config import Config
from metrics import *
import os
from sklearn.utils import shuffle
import math
from moviepy.editor import VideoFileClip

config = Config()
if not os.path.exists('result'):
    os.mkdir('result')
if not os.path.exists('logs'):
    os.mkdir('logs')
if not os.path.exists('checkpoints'):
    os.mkdir('checkpoints')
if not os.path.exists('video_out'):
    os.mkdir('video_out')

#### Training generator, generates augmented images

def generate_train_batch(data, batch_size=32):
    batch_images = np.zeros((batch_size, config.img_rows, config.img_cols, 3))
    batch_masks = np.zeros((batch_size, config.img_rows, config.img_cols, 1))
    while 1:
        for i_batch in range(batch_size):
            i_line = np.random.randint(len(data))
            name_str, img, bb_boxes = get_image_name(df_vehicles, i_line,
                                                     size=(config.img_cols, config.img_rows),
                                                     augmentation=True,
                                                     trans_range=50,
                                                     scale_range=50
                                                     )
            img_mask = get_mask_seg(img, bb_boxes)
            batch_images[i_batch] = img
            batch_masks[i_batch] = img_mask
        yield batch_images, batch_masks

def generate_data_batch(data, dataframe, batch_size=32):
    batch_images = np.zeros((batch_size, config.img_rows, config.img_cols, 3))
    batch_masks = np.zeros((batch_size, config.img_rows, config.img_cols, 1))
    # shuffle data
    # data = shuffle(data)
    loaded_elements = 0
    while loaded_elements < batch_size:
        file_name = data[np.random.randint(len(data))]
        # file_name = data.pop()
        name_str, img, bb_boxes = get_image_name(dataframe, file_name,
                                                   size=(config.img_cols, config.img_rows),
                                                   augmentation=True,
                                                   trans_range=50,
                                                   scale_range=50
                                                   )
        img_mask = get_mask_seg(img, bb_boxes)
        batch_images[loaded_elements] = img
        batch_masks[loaded_elements] = img_mask
        loaded_elements += 1
    yield batch_images, batch_masks


#### Testing generator, generates augmented images
def generate_test_batch(data, batch_size=32):
    batch_images = np.zeros((batch_size, config.img_rows, config.img_cols, 3))
    batch_masks = np.zeros((batch_size, config.img_rows, config.img_cols, 1))
    while 1:
        for i_batch in range(batch_size):
            i_line = np.random.randint(2000)
            i_line = i_line + len(data) - 2000
            name_str, img, bb_boxes = get_image_name(df_vehicles, i_line,
                                                     size=(config.img_cols, config.img_rows),
                                                     augmentation=False,
                                                     trans_range=0,
                                                     scale_range=0
                                                     )
            img_mask = get_mask_seg(img, bb_boxes)
            batch_images[i_batch] = img
            batch_masks[i_batch] = img_mask
        yield batch_images, batch_masks



def get_unet():
    inputs = Input((config.img_rows, config.img_cols, 3))
    inputs_norm = Lambda(lambda x: x / 127.5 - 1.)
    conv1 = Convolution2D(8, 3, 3, activation='relu', border_mode='same')(inputs)
    conv1 = Convolution2D(8, 3, 3, activation='relu', border_mode='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Convolution2D(16, 3, 3, activation='relu', border_mode='same')(pool1)
    conv2 = Convolution2D(16, 3, 3, activation='relu', border_mode='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(pool2)
    conv3 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(pool3)
    conv4 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(pool4)
    conv5 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(conv5)

    up6 = concatenate([UpSampling2D(size=(2, 2))(conv5), conv4], axis=3)
    conv6 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(up6)
    conv6 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv6)

    up7 = concatenate([UpSampling2D(size=(2, 2))(conv6), conv3], axis=3)
    conv7 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(up7)
    conv7 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(conv7)

    up8 = concatenate([UpSampling2D(size=(2, 2))(conv7), conv2], axis=3)
    conv8 = Convolution2D(16, 3, 3, activation='relu', border_mode='same')(up8)
    conv8 = Convolution2D(16, 3, 3, activation='relu', border_mode='same')(conv8)

    up9 = concatenate([UpSampling2D(size=(2, 2))(conv8), conv1], axis=3)
    conv9 = Convolution2D(8, 3, 3, activation='relu', border_mode='same')(up9)
    conv9 = Convolution2D(8, 3, 3, activation='relu', border_mode='same')(conv9)

    conv10 = Convolution2D(1, 1, 1, activation='sigmoid')(conv9)

    model = Model(input=inputs, output=conv10)

    return model

def step_decay(epoch):
   initial_lrate = 1e-4
   drop = 0.5
   epochs_drop = 10.0
   lrate = initial_lrate * math.pow(drop,
           math.floor((1+epoch)/epochs_drop))
   return lrate

def process_video(input_file, output_file):
    """ Given input_file video, save annotated video to output_file """
    # video = VideoFileClip(input_file).subclip(40,44) # from 38s to 46s
    video = VideoFileClip(input_file)
    annotated_video = video.fl_image(process_pipeline)
    annotated_video.write_videofile(output_file, audio=False)

def process_pipeline(frame):
    frame_out = get_BB_new_img(frame, model, verbose=False)
    frame_out = cv2.cvtColor(frame_out, cv2.COLOR_RGB2BGR)

if __name__ == '__main__':


    # Load data
    df_vehicles1 = load_data('object-detection-crowdai', 'labels.csv', verbose=False)
    df_vehicles2 = load_data('object-dataset', 'labels.csv', verbose=False)

    # Concatenate

    df_vehicles = pd.concat([df_vehicles1,df_vehicles2]).reset_index()
    df_vehicles = df_vehicles.drop('index', 1)
    df_vehicles.columns =['File_Path','Frame','Label','ymin','xmin','ymax','xmax']

    print(len(df_vehicles))

    # split udacity csv data into training and validation

    train_data, val_data = split_train_val(df_vehicles)

    ### Generator



    model = get_unet()
    model.summary()
    if config.mode == 'train':
        # training_gen = generate_data_batch(train_data, df_vehicles, config.batch_size)
        training_gen = generate_train_batch(train_data, config.batch_size)

        eval_gen = generate_train_batch(val_data, config.batch_size)

        smooth = 1.

        model.compile(optimizer=Adam(lr=1e-4),
                      loss=IOU_calc_loss, metrics=[IOU_calc])



        # define callbacks to save history and weights
        checkpointer = ModelCheckpoint('checkpoints/weights.{epoch:02d}-{val_loss:.3f}.hdf5')
        logger = CSVLogger(filename='logs/history.csv')
        tflogger = TensorBoard(log_dir='./logs', histogram_freq=0, batch_size=config.batch_size, write_graph=True, write_grads=False,
                                    write_images=False, embeddings_freq=0, embeddings_layer_names=None,
                                    embeddings_metadata=None, embeddings_data=None, update_freq='epoch')
        #learning rate
        # learning schedule callback
        lrate = LearningRateScheduler(step_decay)

        ### Using previously trained data. Set load_pretrained = False, increase epochs and train for full training.
        load_pretrained = True
        if load_pretrained == True:
            model.load_weights("result/model_pretrained.h5")

        # history = model.fit_generator(training_gen,
        #                               samples_per_epoch=1000,
        #                               nb_epoch=1)
        model.fit_generator(generator=training_gen,
                          steps_per_epoch=len(train_data)/config.batch_size,
                          epochs=50,
                          validation_data=eval_gen,
                          validation_steps=len(val_data)/config.batch_size,
                          callbacks=[checkpointer, logger, tflogger, lrate])
    elif config.mode == 'eval':
        eval_gen = generate_train_batch(val_data, config.batch_size)
        model.load_weights("result/model_pretrained.h5")
        ### Test on last frames of data

        batch_img, batch_mask = next(eval_gen)
        pred_all = model.predict(batch_img)
        print(np.shape(pred_all))


    elif config.mode == 'test':
        model.load_weights("result/model_pretrained.h5")
        ### Test on new image
        test_img_dir = 'test_images'
        for test_img in os.listdir(test_img_dir):

            frame = cv2.imread(os.path.join(test_img_dir, test_img))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            frame_out = get_BB_new_img(frame, model, verbose=True)
            frame_out = cv2.cvtColor(frame_out, cv2.COLOR_RGB2BGR)
            cv2.imwrite('output_images/{}'.format(test_img), frame_out)

    elif config.mode == 'video':
        model.load_weights("result/model_pretrained.h5")
        heatmap_prev = np.zeros((640, 960))

        heatmap_10 = [np.zeros((640, 960))] * 10

        video_file = 'project_video.mp4'

        cap_in = cv2.VideoCapture(video_file)
        video_out_dir = 'video_out'

        f_counter = 0
        while True:

            ret, frame = cap_in.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            if ret:

                f_counter += 1

                frame_out = get_BB_new_img(frame, model)
                frame_out = cv2.cvtColor(frame_out, cv2.COLOR_RGB2BGR)

                cv2.imwrite(os.path.join(video_out_dir, '{:06d}.jpg'.format(f_counter)), frame_out)

                cv2.imshow('', frame_out)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        # When everything done, release the capture
        cap_in.release()
        cv2.destroyAllWindows()
        exit()

    print("Done")




