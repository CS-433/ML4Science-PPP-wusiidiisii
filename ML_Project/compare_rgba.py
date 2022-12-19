'''
Author: YourName
Date: 2022-12-18 23:47:39
LastEditTime: 2022-12-19 02:04:25
LastEditors: YourName
Description: 
'''

from PIL import Image
import matplotlib.pyplot as plt
import os

image_path = './ML_Project/prediction/model_output_test_1'
image_pred_path = os.path.join(image_path, 'rgba')
image_true_path = os.path.join(image_path, 'true_rgba')
image_compare_path = os.path.join(image_path, 'compare_rgba')

predict_number = 100

for index in range(predict_number):

    true_file_name = 'test_' + str(index).zfill(5) + '.png'
    pred_file_name = 'bg' + str(int(index/2)).zfill(4) + '_rgba_' + str(index).zfill(5) + '.png'

    img1 = Image.open(os.path.join(image_true_path, true_file_name))
    img2 = Image.open(os.path.join(image_pred_path, pred_file_name))
    result = Image.new(img1.mode, (280*2, 256))
    result.paste(img1, box=(0, 0))
    result.paste(img2, box=(280, 0))

    if index % 9 == 0:
        result_combine = Image.new(img1.mode, (280*2*3, 280*3))
    result_combine.paste(img1, box=((index%3)*560, int((index%9/3))*280))
    result_combine.paste(img2, box=((index%3)*560+280, int((index%9)/3)*280))

    compare_file_name = 'compare_' + str(index).zfill(5) + '.png'
    result.save(os.path.join(image_compare_path, compare_file_name))

    compare_file_name = 'compare_combine_' + str(int(index/9)).zfill(5) + '.png'
    if index % 9 == 8:
        result_combine.save(os.path.join(image_compare_path, compare_file_name))

