## 比赛信息
- 本次大赛要求选手开发算法模型，通过布样影像，基于对布样中疵点形态、长度、面积以及所处位置等的分析，判断瑕疵的种类 。通过探索布样疵点精确智能诊断的优秀算法，提升布样疵点检验的准确度，降低对大量人工的依赖，提升布样疵点质检的效果和效率。
- [比赛链接](https://tianchi.aliyun.com/competition/information.htm?spm=5176.100067.5678.2.70731756uJzvoZ&raceId=231666)

## 文件说明
- src : 存放所有相关代码的文件夹
    - main.py : finetune**主函数**
    - extract_pic_xml.py : 将所有原始数据的xml提取整合到目标xml文件夹中,pic提取整合到pic文件夹中
    - cname2ename.py :  包含中文名和拼音的dict
    - split_10.py : 将原始数据按照比赛要求分成10个文件夹, 取一部分作为validation数据另外存放
    - DataAugmentForObejctDetection.py : 数据增强脚本
    - DataAugForTrainAndTest.py ： 训练数据增强
    - xml_helper.py : 处理xml文件的辅助脚本
    - cp_to_train.py : 将data_split_train中原图copy到data_for_train文件夹下面,将增强后的图移动到	   data_for_train文件夹下面
    - merge.py : 融合脚本
- models : 存放预训练模型以及训练输出的模型
- data : 存放训练数据以及test数据
- submit : 存放预测的中间结果
- result.csv :　根据submit中的结果文件进行融合后的最后结果
- run.sh : 复现线上结果的脚本(需下载我预先训练好的模型,同时在main函数中注释掉训练即测试部分,取消复现部分代码)
- run_all.sh : 从头开始训练并得到预测结果的脚本。

## 操作说明
- step1 : 手动解压原始数据压缩文件，得保证解压后的文件名没有乱码！！！！！！！
- step2 : **手动在keras源码中修改插值方式**：
    - 在~/anaconda3/envs/tianchi/lib/python2.7/site-packages/keras_preprocessing/imge.py（imge.py路径）的第33行后面加入 'antialias': pil_image.ANTIALIAS
    - 变成：
        '''
	_PIL_INTERPOLATION_METHODS = {
		'nearest': pil_image.NEAREST,
		'bilinear': pil_image.BILINEAR,
		'bicubic': pil_image.BICUBIC,
		'antialias': pil_image.ANTIALIAS,   #added by mao
	    }
        '''
- step3 : 运行run.sh可以复现线上的结果（需要先下载[训练好的模型](https://pan.baidu.com/s/1DEcTbu_oifw6L5eqK7JVnA) 密码: qp9k）；或者运行run_all.sh可以从头开始训练并得到预测结果。

## 思路说明
- 数据输入部分：
    - 输入:
    	1. 图片大小：H,W=[550, 600]
    	2. batch : 4
    - 数据增强线下：每张瑕疵图片扩充到三到四张，加入了裁剪，改变亮度，加噪声，cutout等方式，当然增强的时候利用了xml文件的信息，保证了框也随之变化
    - 数据增强线上：使用keras内置的增强方法，开启了镜像
- 模型及trick部分：
    - 模型：densenet169
    - trick：
	1. 使用LSR(Label smoothing Regularization)损失，超参e={0.25,0.3}。增加模型泛化能力。
	2. 尝试balanced CE　以及　balanced CE + focal loss损失，但是均未带来提升。
	3. 尝试增加attetion，也未带来提升。
	4. 模型融合.单模730, 双模融合746。
- 训练策略：
	- 每个类别的10%图片作为validation集合
	- 剩下90%中只对带瑕疵图片进行图像增强。每张扩充到3到4张。
- 输出结果说明：
    - 输出模型存放位置：./models/final
    - 输出模型名称：1) model_weight_0.7296.hdf5; 2) model_weight_0.7239.hdf5
    - 输出模型大小：均为102.5Mb
- 另：
	- 单模730的参数：LSR损失的e=0.25, 图片增强的时候每张扩充到3张
  - 单模72+的参数：LSR损失的e=0.3, 图片增强的时候每张扩充到4张
  - 其他参数两者均一致，具体间main.py函数
- **注意，模型初始化用的是imagenet预训练权值，在开始模型训练前会自行下载**

## 随机性
- 线下线上增强是随机增强的，这会有一个随机性，重新训练的话结果可能会在线上最好成绩附近波动

## 初赛和复赛的数据链接: 
- [初赛part1](https://pan.baidu.com/s/1KoZcXKCCaWLWfGc5Q4gCjg), 密码: 2qdn
- [初赛part2](https://pan.baidu.com/s/1c0o7WKm-ETPcIyF6JPS3Wg),  密码: jq9a
- [复赛](https://pan.baidu.com/s/1wuA0VT7E7SBtkrvarfPCcw), 密码: vyj9

## 另
- 初赛代码链接: [maozezhong/TIANCHI_XUELANG_AI](https://github.com/maozezhong/TIANCHI_XUELANG_AI)
- 相关工具链接: [maozezhong/CV_ToolBox](https://github.com/maozezhong/CV_ToolBox)
- [maozezhong/focal_loss_multi_class](https://github.com/maozezhong/focal_loss_multi_class)
