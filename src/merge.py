# -*- coding=utf-8 -*-

import pandas as pd
import os

path = '../submit'
file_dict = dict()
final_res_path = '../result.csv'
cnt = 0
w_list = [0.7296, 0.7239]
w_list = [w/sum(w_list) for w in w_list]
for parent, _, files in os.walk(path):
    for file in files:
        data_path = os.path.join(parent, file)
        data = pd.read_csv(data_path)
        for i in range(len(data['filename|defect'])):
            file_name = data['filename|defect'][i]
            pro = data['probability'][i]*w_list[cnt]
            if file_name in file_dict.keys():
                file_dict[file_name] += pro
            else:
                file_dict[file_name] = pro
        cnt += 1

keys = file_dict.keys()
sorted(keys)

file_name_list = list()
pro_list = list()
for key in keys:
    file_name_list.append(key)
    pro = file_dict[key]
    pro = round(pro, 6)
    pro = max(0.000001, pro)
    pro = min(0.999999, pro)
    pro_list.append(pro)

dataframe = pd.DataFrame({'filename': file_name_list, 'probability': pro_list})
dataframe.to_csv(final_res_path, index=False, header=True)