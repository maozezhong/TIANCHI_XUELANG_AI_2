## 文件说明
- extract_pic_xml.py : 将所有原始数据的xml提取整合到目标xml文件夹中,pic提取整合到pic文件夹中
- cname2ename.py :  包含中文名和拼音的dict
- split_10.py : 将原始数据按照比赛要求分成10个文件夹, 取一部分作为validation数据另外存放
- DataAugmentForObejctDetection.py : 数据增强脚本
- DataAugForTrainAndTest.py ： 训练数据增强
- xml_helper.py : 处理xml文件的辅助脚本
- cp_to_train.py : 将data_split_train中原图copy到data_for_train文件夹下面,将增强后的图移动到data_for_train文件夹下面
- main.py : finetune主函数
- merge.py : 融合脚本

## 数据处理步骤
- 运行extract_pic_xml.py， 将所有图片及xml文件分别整合到一起
- 运行split_10.py : 根据xml文件信息将所有图片按照官网要求分成10个文件夹, 并抽取一部分作为validation数据
- 运行DataAugForTrainAndTest.py
- 运行cp_to_train.py
---
- 或者直接运行data_process.sh

## 主函数
- 运行main.py
- 运行merge.py

