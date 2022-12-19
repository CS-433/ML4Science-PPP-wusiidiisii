'''
Author: YourName
Date: 2022-12-18 22:33:59
LastEditTime: 2022-12-19 01:37:44
LastEditors: YourName
Description: 
'''
import os
import json
import numpy as np

test_txt_path = './ML_Project/prediction/test_list.txt'

test_file_list = []
with open(test_txt_path, "r") as f:  # 打开文件
    data = f.readlines()  # 读取文件
    for ln in data:
        test_file_list.append(ln.strip('\n'))
f.close()

true_file_path = './ML_Project/output/dataset'
model_output_path = './ML_Project/prediction/model_output_test_1/model_output'
pred_file_path = './ML_Project/prediction/model_output_test_1/matrix'
bg_index = 0
test_file_index = 0
for test_file_number in test_file_list:
    test_files = test_file_number.split(' ')
    for test_file in test_files:
        temp_path = 'bg' + str(bg_index).zfill(4)
        single_true_file_path = os.path.join(true_file_path, temp_path)
        single_true_file = os.path.join(single_true_file_path, 'matrix')
        temp_path = 'bg' + str(bg_index).zfill(4) + '_extrinsic_' + str(test_file).zfill(5) + '.json'
        single_true_file = os.path.join(single_true_file, temp_path)
        with open(single_true_file, 'r') as f:
            data_single_true = json.load(f)
        f.close()
        
        temp_path = str(test_file_index).zfill(5) + '.json'
        single_pred_file = os.path.join(model_output_path, temp_path)

        with open(single_pred_file, 'r') as f:
            data_single_pred = json.load(f)
        f.close()
        data_single_pred = np.array(data_single_pred)
        add_array = np.array([[0, 0, 0, 1]])
        data_single_pred = np.r_[data_single_pred, add_array]

        data_single_true['matrix_probe_in_cam_coord_pre'] = (data_single_pred).tolist()

        # pred_output_file_name = 'bg' + str(bg_index).zfill(4) + '_extrinsic_' + str(test_file).zfill(5) + '.json'
        pred_output_file_name = 'prediction_label' + str(test_file_index).zfill(4) + '.json'
        pred_output_json = json.dumps(data_single_true, indent=4)
        with open(os.path.join(pred_file_path, pred_output_file_name), 'w') as json_file:
            json_file.write(pred_output_json)
        f.close()
        test_file_index += 1

    bg_index += 1