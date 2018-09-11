# coding=utf-8
#################################################
# finetune
#################################################
import pandas as pd
import numpy as np
import os
import random
import cv2
import datetime
from tqdm import tqdm

from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from keras.models import Model
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers.pooling import GlobalAveragePooling2D
from keras.optimizers import Adam, SGD

from keras.applications.densenet import DenseNet169, DenseNet121
from keras.applications.densenet import preprocess_input

from focal_loss import focal_loss
from attetion import add_attention

#######################################
# 在训练的时候置为1
from keras import backend as K
K.set_learning_phase(1)
#######################################

EPOCHS = 60
RANDOM_STATE = 2018
learning_rate = 0.001#0.001

TRAIN_DIR = '../data/data_for_train'
VALID_DIR = '../data/data_for_valid'    

# 获得每一类的图片张数，索引顺序与keras中的读取顺序对应
def get_cls_num(directory):
    '''
    '''
    classes = list()
    for subdir in sorted(os.listdir(directory)):
        if os.path.isdir(os.path.join(directory, subdir)):
            classes.append(subdir)
    
    classes_num = list()
    for cls_ in classes:
        path = os.path.join(directory, cls_)
        pics = os.listdir(path)
        classes_num.append(len(pics))
    
    return classes_num

def get_classes_indice(directory):
    '''
    得到索引与label对应的dict
    '''
    classes = []
    for subdir in sorted(os.listdir(directory)):
        if os.path.isdir(os.path.join(directory, subdir)):
            classes.append(subdir)
    num_classes = len(classes)
    class_indices = dict(zip(classes, range(len(classes))))
    return class_indices

def get_callbacks(filepath, patience=2):
    lr_reduce = ReduceLROnPlateau(monitor='val_acc', factor=0.1, epsilon=5e-6, patience=patience, verbose=1, min_lr = 0.00001)
    msave = ModelCheckpoint(filepath, monitor='val_acc', save_best_only=True) #该回调函数将在每个epoch后保存模型到filepath
    earlystop = EarlyStopping(monitor='val_acc', min_delta=0, patience=patience*3+4, verbose=1, mode='auto')
    return [lr_reduce, msave, earlystop]

def add_new_last_layer(base_model, nb_classes, drop_rate=0.):
    """Add last layer to the convnet
    Args:
        base_model: keras model excluding top
        nb_classes: # of classes
    Returns:
        new keras model with last layer
    """
    bn_axis = 3 if K.image_data_format() == 'channels_last' else 1
    x = base_model.output
    #x = BatchNormalization(axis=bn_axis, epsilon=1.001e-5,name='bn_final')(x)
    x = Dropout(0.5)(x)
    x = GlobalAveragePooling2D()(x) 
    predictions = Dense(nb_classes, activation='softmax')(x) #new softmax layer
    model = Model(input=base_model.input, output=predictions)
    return model

# LSR(label smoothing regularization)
def mycrossentropy(e = 0.25,nb_classes=11):
    '''
        https://spaces.ac.cn/archives/4493
    '''
    def mycrossentropy_fixed(y_true, y_pred):
        return (1-e)*K.categorical_crossentropy(y_true,y_pred) + e*K.categorical_crossentropy(K.ones_like(y_pred)/nb_classes, y_pred)
    return mycrossentropy_fixed

# LSR(label smoothing regularization) + focal_loss
def myloss(classes_num, e = 0.25,nb_classes=11):
    def myloss_fixed(y_true, y_pred):
        
        import tensorflow as tf
        from tensorflow.python.ops import array_ops
        from keras import backend as K

        zeros = array_ops.zeros_like(y_pred, dtype=y_pred.dtype)
        crossentropy_loss = -1 * y_true * tf.log(tf.clip_by_value(y_pred, 1e-8, 1.0))

        classes_weight = array_ops.zeros_like(y_true, dtype=y_true.dtype)

        total_num = float(sum(classes_num))
        classes_w_t1 = [ total_num / ff for ff in classes_num ]
        sum_ = sum(classes_w_t1)
        classes_w_t2 = [ ff/sum_ for ff in classes_w_t1 ]   #scale
        classes_w_tensor = tf.convert_to_tensor(classes_w_t2, dtype=y_true.dtype)
        classes_weight += classes_w_tensor

        alpha = array_ops.where(y_true > zeros, classes_weight, zeros)

        balanced_fl = alpha * crossentropy_loss
        balanced_fl = tf.reduce_sum(balanced_fl)

        return (1-e)*balanced_fl + e*K.categorical_crossentropy(K.ones_like(y_pred)/nb_classes, y_pred)
    return myloss_fixed

def get_model(INT_HEIGHT, IN_WIDTH):
    '''
    获得模型
    '''

    base_model = DenseNet169(include_top=False, weights='imagenet', input_shape=(INT_HEIGHT,IN_WIDTH, 3))
    #base_model = DenseNet169(include_top=False, weights='../save_model/densenet169_weights_tf_dim_ordering_tf_kernels_notop.h5', input_shape=(INT_HEIGHT,IN_WIDTH, 3))

    # add attention
    # base_model = add_attention(base_model)
    
    model = add_new_last_layer(base_model, 11)
    model.compile(optimizer=SGD(lr=learning_rate, momentum=0.9), loss=[mycrossentropy()], metrics=['accuracy'])
    # classes_num = get_cls_num('../data/data_for_train')
    # model.compile(optimizer=SGD(lr=learning_rate, momentum=0.9), loss=[myloss(classes_num)], metrics=['accuracy'])
    model.summary()

    return model

def train_model(save_model_path, BATCH_SIZE, IN_SIZE):

    INT_HEIGHT = IN_SIZE[0]
    IN_WIDTH = IN_SIZE[1]

    callbacks = get_callbacks(filepath=save_model_path, patience=3)
    model = get_model(INT_HEIGHT, IN_WIDTH)

    train_datagen = ImageDataGenerator(
        preprocessing_function = preprocess_input,  
        horizontal_flip = True, 
        vertical_flip = True
    )

    valid_datagen = ImageDataGenerator(
        preprocessing_function = preprocess_input
    )

    train_generator = train_datagen.flow_from_directory(
        directory = TRAIN_DIR, 
        target_size = (INT_HEIGHT, IN_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        seed=RANDOM_STATE,
        interpolation='antialias',  # PIL默认插值下采样的时候会模糊
    )

    valid_generator = valid_datagen.flow_from_directory(
        directory = VALID_DIR,
        target_size = (INT_HEIGHT, IN_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        seed=RANDOM_STATE,
        interpolation='antialias',  # PIL默认插值下采样的时候会模糊
    )

    model.fit_generator(
        train_generator, 
        steps_per_epoch = 2*(train_generator.samples // BATCH_SIZE + 1), 
        epochs = EPOCHS,
        max_queue_size = 100,
        workers = 1,
        verbose = 1,
        validation_data = valid_generator, #valid_generator,
        validation_steps = valid_generator.samples // BATCH_SIZE, #valid_generator.samples // BATCH_SIZE + 1, #len(valid_datagen)+1, 
        callbacks = callbacks
        )

def predict(weights_path, IN_SIZE):
    '''
    对测试数据进行预测
    '''

    INT_HEIGHT = IN_SIZE[0]
    IN_WIDTH = IN_SIZE[1]
    K.set_learning_phase(0)

    test_pic_root_path = '../data'

    filename = []
    probability = []

    #1#得到模型
    model = get_model(INT_HEIGHT, IN_WIDTH)
    model.load_weights(weights_path)

    #2#预测
    for parent,_,files in os.walk(test_pic_root_path):
        for line in tqdm(files):

            pic_path = os.path.join(test_pic_root_path, line.strip())

            img = load_img(pic_path, target_size=(INT_HEIGHT, IN_WIDTH, 3), interpolation='antialias')
            img = img_to_array(img)
            img = preprocess_input(img)
            img = np.expand_dims(img, axis=0)

            prediction = model.predict(img)[0] 
            prediction = list(prediction)
            
            classes_dict = get_classes_indice(TRAIN_DIR)
            classes_dict = {classes_dict[key]:key for key in classes_dict.keys()}
            
            for index, pro in enumerate(prediction):
                # 获得对应label名称
                label = classes_dict[index] 
                pro = round(pro, 5)
                if pro == 0:
                    pro = 0.00001
                if pro == 1:
                    pro = 0.99999
                
                #存入list
                filename.append(line.strip()+'|'+label)
                probability.append(' '+str(pro))
            
    #3#写入csv
    res_path = "../submit/submit_"+datetime.datetime.now().strftime('%Y%m%d_%H%M%S') + ".csv"
    dataframe = pd.DataFrame({'filename|defect': filename, 'probability': probability})
    dataframe.to_csv(res_path, index=False, header=True)

def main():

    #～～～～～～～～～～～～～～～～～  复现线上最优结果　模型预测 ～～～～～～～～～～～～～～～～～

    batch_size = 4
    in_size = [550,600] #[H,W]

    weights_path1 = '../models/final/model_weight_0.7296.hdf5'  #dense169, [550,600]    submit_1.csv
    weights_path2 = '../models/final/model_weight_0.7239.hdf5'   #dense169, [550,600]    submit_2.csv
 
    ## 预测得到线上746的结果
    predict(weights_path=weights_path1, IN_SIZE=in_size)
    predict(weights_path=weights_path2, IN_SIZE=in_size)


    #～～～～～～～～～～～～～～～～～  训练及预测全过程 ～～～～～～～～～～～～～～～～～
    
    # weights_path = '../models/model_weight.hdf5'
    # train_model(save_model_path=weights_path, BATCH_SIZE=batch_size, IN_SIZE=in_size)  
    # predict(weights_path=weights_path, IN_SIZE=in_size)
    
if __name__=='__main__':
    main()
