# -*- coding=utf-8 -*-
##############################################################
# 对训练集进行数据增强
# 增强后的数据作为validation集合
##############################################################

import time
import random
import cv2
import os
import math
import numpy as np
from skimage.util import random_noise
from skimage import exposure

def show_pic(img, bboxes=None):
    '''
    输入:
        img:图像array
        bboxes:图像的所有boudning box list, 格式为[[x_min, y_min, x_max, y_max]....]
        names:每个box对应的名称
    '''
    cv2.imwrite('./1.jpg', img)
    img = cv2.imread('./1.jpg')
    for i in range(len(bboxes)):
        bbox = bboxes[i]
        x_min = bbox[0]
        y_min = bbox[1]
        x_max = bbox[2]
        y_max = bbox[3]
        cv2.rectangle(img,(int(x_min),int(y_min)),(int(x_max),int(y_max)),(0,255,0),3) 
    cv2.namedWindow('pic', 0)  # 1表示原图
    cv2.moveWindow('pic', 0, 0)
    cv2.resizeWindow('pic', 1200,800)  # 可视化的图片大小
    cv2.imshow('pic', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows() 
    os.remove('./1.jpg')

import shutil
from xml_helper import *
from tqdm import tqdm
from DataAugmentForObejctDetection import DataAugmentForObjectDetection

def data_aug(source_pic_root_path, source_xml_root_path, target_pic_root_path, aug_ob, need_aug_num=1, add_norm=False):
    '''
    输入：
        source_pic_root_path ： 源图片根目录
        source_xml_root_path ： 源xml根目录
        target_pic_root_path ： 目标图片根目录
        aug_ob : 增强类的对象
        need_aug_num ： 需要每张图片增强的张数
    '''

    for parent, _, files in os.walk(source_pic_root_path):
        # random.shuffle(files)
        for file in tqdm(files):
            if not file[-3:] == 'jpg':
                continue
            if not add_norm and parent.split('/')[-1] == 'norm':
                continue
            auged_num = 0                       #当前已经增强的次数计数
            while auged_num < need_aug_num:
                auged_num += 1
                pic_path = os.path.join(parent, file)
                xml_path = os.path.join(source_xml_root_path, file[:-4]+'.xml')
                
                img = cv2.imread(pic_path)
                h,w,_ = img.shape

                if os.path.exists(xml_path):    #'bad'的数据才有xml
                    coords = parse_xml(xml_path)        #解析得到box信息，格式为[[x_min,y_min,x_max,y_max,name]]
                    coords = [coord[:4] for coord in coords]
                else:
                    coords = [[0,0,w,h]]  #如果是没有框的图片，即无瑕疵的布匹图片，就给它一个整体图像的框，即不进行crop了

                # show_pic(img, coords)    # 原图

                auged_img, auged_bboxes = aug_ob.dataAugment(img, coords)

                # show_pic(auged_img, auged_bboxes)  # 强化后的图

                temp_target_pic_root_path = os.path.join(target_pic_root_path, parent.split('/')[-1])
                if not os.path.exists(temp_target_pic_root_path):
                    os.mkdir(temp_target_pic_root_path)
                target_pic_path = os.path.join(temp_target_pic_root_path, file[:-4]+'_aug'+str(auged_num)+'.jpg')
                cv2.imwrite(target_pic_path, auged_img) #写入增强图片

if __name__ == '__main__':
    
    # 训练增强只进行：裁剪，改亮度，加噪，cutout
    dataAug_train = DataAugmentForObjectDetection(rotation_rate=0,
                crop_rate=0.5, shift_rate=0, change_light_rate=0.5,
                add_noise_rate=0.5, flip_rate=0, cutout_rate=0.5)   
    # 验证增强只进行：改亮度，加噪，翻转
    dataAug_valid = DataAugmentForObjectDetection(rotation_rate=0,
                crop_rate=0, shift_rate=0, change_light_rate=0.5,
                add_noise_rate=0.5, flip_rate=0.5, cutout_rate=0) 
    
    source_pic_root_path = '../data/data_split_train'
    source_xml_root_path = '../data/xml'

    #1# 对训练数据增强
    print('~~~~~~~~~~~~ 对训练集进行增强 ~~~~~~~~~~~~')
    target_pic_root_path = '../data/data_AugForTrain'
    if os.path.exists(target_pic_root_path):
        shutil.rmtree(target_pic_root_path)
    os.makedirs(target_pic_root_path)

    data_aug(source_pic_root_path, source_xml_root_path, target_pic_root_path, dataAug_train, need_aug_num=2)

    # #2# 增强后的图片作为测试集
    # print('~~~~~~~~~~~~ 对原图增强作为valid ~~~~~~~~~~~~')
    # target_pic_root_path = '../data/data_for_valid'
    # if os.path.exists(target_pic_root_path):
    #     shutil.rmtree(target_pic_root_path)
    # os.makedirs(target_pic_root_path)

    # data_aug(source_pic_root_path, source_xml_root_path, target_pic_root_path, dataAug_valid, add_norm=True)

    



