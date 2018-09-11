# -*- coding=utf-8 -*-
'''
讲所有xml文件copy到../data/xml文件夹下
将所有pic文件copy到../data/pics文件夹下
'''
import os
import shutil
from tqdm import tqdm

def extract_pic_xml(source_root_path, target_xml_root_path, target_pic_root_path):
    '''
    输入：
        source_root_path : 原始存放有xml的文件夹根目录
        target_xml_root_path : 目标xml文件夹根目录
        target_pic_root_path : 目标pic文件夹根目录
    '''

    for parent, _, files in os.walk(source_root_path):
        for file in tqdm(files):
            file_name = os.path.join(parent, file)
            if file_name[-3:] == 'xml':
                shutil.copyfile(file_name, os.path.join(target_xml_root_path, file))
            elif file_name[-3:] =='jpg':
                shutil.copyfile(file_name, os.path.join(target_pic_root_path, file))

if __name__ == '__main__':

    source_root_path = '../data/data_ori_train'
    target_xml_root_path = '../data/xml'
    target_pic_root_path = '../data/pics'
    if os.path.exists(target_xml_root_path):
        shutil.rmtree(target_xml_root_path)
    if os.path.exists(target_pic_root_path):
        shutil.rmtree(target_pic_root_path)
    os.makedirs(target_xml_root_path)
    os.makedirs(target_pic_root_path)

    extract_pic_xml(source_root_path, target_xml_root_path, target_pic_root_path)
