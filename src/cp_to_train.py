# -*- coding=utf-8 -*-

import os
import shutil
from tqdm import tqdm

source_pic_root = '../data/data_split_train'
aug_pic_root = '../data/data_AugForTrain'
target_pic_root = '../data/data_for_train'
if os.path.exists(target_pic_root):
    shutil.rmtree(target_pic_root)
os.makedirs(target_pic_root)

#1# 复制原始图片到目标位置
for parent, _, files in os.walk(source_pic_root):
    for file in tqdm(files):
        if not file[-3:] == 'jpg':  # 不是图片则跳过
            continue
        temp_target_pic_root = os.path.join(target_pic_root, parent.split('/')[-1])
        if not os.path.exists(temp_target_pic_root):
            os.makedirs(temp_target_pic_root)
        shutil.copyfile(os.path.join(parent, file), os.path.join(temp_target_pic_root, file))

#2# 复制增强后的图片到目标位置
for parent, _, files in os.walk(aug_pic_root):
    for file in tqdm(files):
        if not file[-3:] == 'jpg':  # 不是图片则跳过
            continue
        temp_target_pic_root = os.path.join(target_pic_root, parent.split('/')[-1])
        if not os.path.exists(temp_target_pic_root):
            os.makedirs(temp_target_pic_root)
        shutil.move(os.path.join(parent, file), os.path.join(temp_target_pic_root, file))
