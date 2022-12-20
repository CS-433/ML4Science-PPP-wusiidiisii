"""
This file is used to merge dataset of different background
And random split the whole set into train, val and test set
"""
import os
import shutil
import random

# paths and variable initialization
root = "D:\\Code\\dataset_new"
path_train_matrix = "D:\\Code\\ML_dataset_new\\train\\matrix"
path_train_rgba = "D:\\Code\\ML_dataset_new\\train\\rgba"
path_train_list = "D:\\Code\\ML_dataset_new\\train\\list\\list.txt"
path_val_matrix = "D:\\Code\\ML_dataset_new\\val\\matrix"
path_val_rgba = "D:\\Code\\ML_dataset_new\\val\\rgba"
path_val_list = "D:\\Code\\ML_dataset_new\\val\\list\\list.txt"
path_test_matrix = "D:\\Code\\ML_dataset_new\\test\\matrix"
path_test_rgba = "D:\\Code\\ML_dataset_new\\test\\rgba"
path_test_list = "D:\\Code\\ML_dataset_new\\test\\list\\list.txt"
count_train = 0
count_val = 0
count_test = 0
train_idx_list = []
val_idx_list = []
test_idx_list = []
seed_list = []

# use different seed for every random
for i in range(50):
	seed = int(random.random()*1000)
	seed_list.append(seed)

for i, foldername in enumerate(os.listdir(root)):
	path_i = os.path.join(root, "bg{0:04d} (1)".format(i))
	path_i_matrix = os.path.join(path_i, "matrix (1)")
	path_i_rgba = os.path.join(path_i, "rgba (1)")

	# each background folder has 25 images
	# random split into:
	# train:20, val:3, test:2
	random.seed(seed_list[i])
	val_list = random.sample(range(25), 5)
	test_list = random.sample(val_list, 2)
	train_list = list(set([*range(25)]) - set(val_list))
	val_list = list(set(val_list) - set(test_list))

	train_str = ' '.join(map(str, sorted(train_list)))
	val_str = ' '.join(map(str, sorted(val_list)))
	test_str = ' '.join(map(str, sorted(test_list)))
	train_idx_list.append(train_str)
	val_idx_list.append(val_str)
	test_idx_list.append(test_str)

	# rename the files and put into new directories
	for j, filename in enumerate(os.listdir(path_i_matrix)):
		name_old_matrix = os.path.join(path_i_matrix, "bg{0:04d}_extrinsic_{1:05d} (1).json".format(i,j))
		name_old_rgba = os.path.join(path_i_rgba, "bg{0:04d}_rgba_{1:05d} (1).png".format(i,j))

		if j in val_list:
			name_new_matrix = os.path.join(path_i_matrix, "val_{0:05d}.json".format(count_val))
			name_new_rgba = os.path.join(path_i_rgba, "val_{0:05d}.png".format(count_val))
			os.rename(name_old_matrix, name_new_matrix)
			os.rename(name_old_rgba, name_new_rgba)
			shutil.move(name_new_matrix, path_val_matrix)
			shutil.move(name_new_rgba, path_val_rgba)
			count_val += 1
		elif j in test_list:
			name_new_matrix = os.path.join(path_i_matrix, "test_{0:05d}.json".format(count_test))
			name_new_rgba = os.path.join(path_i_rgba, "test_{0:05d}.png".format(count_test))
			os.rename(name_old_matrix, name_new_matrix)
			os.rename(name_old_rgba, name_new_rgba)
			shutil.move(name_new_matrix, path_test_matrix)
			shutil.move(name_new_rgba, path_test_rgba)
			count_test += 1
		else:
			name_new_matrix = os.path.join(path_i_matrix, "{0:05d}.json".format(count_train))
			name_new_rgba = os.path.join(path_i_rgba, "{0:05d}.png".format(count_train))
			os.rename(name_old_matrix, name_new_matrix)
			os.rename(name_old_rgba, name_new_rgba)
			shutil.move(name_new_matrix, path_train_matrix)
			shutil.move(name_new_rgba, path_train_rgba)
			count_train += 1

# keep a record of which image goes into which set
with open(path_train_list, 'w') as f_train:
	for line in train_idx_list:
		f_train.write(line)
		f_train.write('\n')
f_train.close()
with open(path_val_list, 'w') as f_val:
	for line in val_idx_list:
		f_val.write(line)
		f_val.write('\n')
f_val.close()
with open(path_test_list, 'w') as f_test:
	for line in test_idx_list:
		f_test.write(line)
		f_test.write('\n')
f_test.close()