# -*- coding=utf-8 -*-
'''
根据原始数据data
得到分类的数据data_split
'''

import os
import shutil
from tqdm import tqdm
from cname2ename import cname_ename
import xml.etree.ElementTree as ET
import random

out = open('./repeat_main_object.txt','w')
def get_name_from_xml(xml_path, target_name_dict):
    '''
    输入：
        xml_path : xml文件
        target_name_dict : 目标文件夹名称dict,不包括其他
    输出：
        对应瑕疵的名称
    '''
    tree = ET.parse(xml_path)		
    root = tree.getroot()
    objs = root.findall('object')
    names = set()
    for ix, obj in enumerate(objs):
        name = obj.find('name').text
        names.add(name.encode('utf-8'))
    names = list(names)
    final_name = names[0]
    repeat_cnt = 0
    for name in names:
        try:
            temp_name = cname_ename[name]
        except:
            print(name)
        if temp_name in target_name_dict.keys():
            repeat_cnt += 1
            final_name = name
    if repeat_cnt > 1:
        out.write(xml_path+'\n')
    return final_name


def split(source_pic_root_path, source_xml_root_path, target_pic_root_path, target_name_dict):
    '''
    输入：
        source_pic_root_path : 包含所有图片的源文件根目录
        source_xml_root_path ： 包含所有xml文件的源文件根目录
        target_pic_root_path ： 目标图片根目录
        target_name_dict : 目标文件夹名称dict,不包括其他
    '''

    for parent, _, files in os.walk(source_pic_root_path):
        for file in tqdm(files):
            pic_path = os.path.join(parent, file)
            xml_path = os.path.join(source_xml_root_path, file[:-4]+'.xml')

            if not os.path.exists(xml_path):            #说明是无瑕疵图片，只有无瑕疵图片没有xml文件
                temp_name = '正常'
            else:
                temp_name = get_name_from_xml(xml_path, target_name_dict) #得到该图片对应的瑕疵名称

            temp_name = cname_ename[temp_name]          #中文名转为英文名

            if temp_name in target_name_dict.keys():
                target_pic_folder = os.path.join(target_root_path, target_name_dict[temp_name])
                if not os.path.exists(target_pic_folder):
                    os.makedirs(target_pic_folder)
                target_pic_path = os.path.join(target_pic_folder, file)
            else:
                target_pic_folder = os.path.join(target_root_path, 'defect_10')
                if not os.path.exists(target_pic_folder):
                    os.makedirs(target_pic_folder) 
                target_pic_path = os.path.join(target_pic_folder, file)
            shutil.copyfile(pic_path, target_pic_path)

def train_test_split(root_path, target_valid_root, rate=0.1):
    '''
    输入:
        root_path : 分类好的train图片文件存放路径
        target_valid_root : validaiton数据存放的根目录
        rate : 作为验证集的数据比例
    '''

    if os.path.exists(target_valid_root):
        shutil.rmtree(target_valid_root)

    for parent, _, files in os.walk(root_path):
        random.shuffle(files)   #先打乱
        limit = int(rate*len(files))
        cnt = 0
        for file in files:
            cnt += 1
            if cnt > limit:
                break
            source_pic_path = os.path.join(parent, file)
            target_pic_path = os.path.join(target_valid_root+'/'+parent.split('/')[-1], file)

            if not os.path.exists(target_valid_root+'/'+parent.split('/')[-1]):
                os.makedirs(target_valid_root+'/'+parent.split('/')[-1])

            shutil.move(source_pic_path, target_pic_path)

if __name__ == '__main__':

    target_name_dict = {'zhengchang':'norm', 'zhadong':'defect_1', 
                        'maoban':'defect_2', 'cadong':'defect_3', 
                        'maodong':'defect_4', 'zhixi':'defect_5', 
                        'diaojing':'defect_6', 'quejing':'defect_7', 
                        'tiaohua':'defect_8', 'youzi':'defect_9','wuzi':'defect_9'}
    
    ## 加入了边扎洞，边缺经，经跳花
    # target_name_dict = {'zhengchang':'norm', 'zhadong':'defect_1', 'bianzhadong' : 'defect_1',
    #                     'maoban':'defect_2', 'cadong':'defect_3', 
    #                     'maodong':'defect_4', 'zhixi':'defect_5', 
    #                     'diaojing':'defect_6', 'quejing':'defect_7', 'bianquejing' : 'defect_7',
    #                     'tiaohua':'defect_8', 'jingtiaohua' : 'defect_8',
    #                     'youzi':'defect_9','wuzi':'defect_9'}

    source_pic_root_path = '../data/pics'
    source_xml_root_path = '../data/xml'
    target_root_path = '../data/data_split_train'
    if os.path.exists(target_root_path):
        shutil.rmtree(target_root_path)

    # 将原始数据按照类别分好文件夹
    split(source_pic_root_path, source_xml_root_path, target_root_path, target_name_dict)

    # 分出一部分作为验证集和训练集
    print('~~~~~~~~~~~~~ train_test_split ~~~~~~~~~~~~~')
    train_test_split(target_root_path, '../data/data_for_valid', rate=0.1)
