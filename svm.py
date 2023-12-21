from libsvm.svmutil import *
import numpy as np
import matplotlib.pyplot as plt

y_train, x_train = svm_read_problem('a9a.dat')
y_test, x_test = svm_read_problem('a9at.dat')


# use ex4 data
def convert_to_libsvm_format(x_file, y_file, output_file):
    with open(x_file, 'r') as fx, open(y_file, 'r') as fy, open(output_file, 'w') as fout:
        for x_line, y_line in zip(fx, fy):
            features = x_line.strip().split()
            modified_features = [str(round(float(value[:4]) * 10, 1)) for value in features]
            label = y_line.strip()[0]

            libsvm_line = label
            for i, value in enumerate(modified_features, start=1):
                libsvm_line += f" {i}:{value}"

            fout.write(libsvm_line + "\n")


# convert_to_libsvm_format("ex4x.dat", "ex4y.dat", "ex4.dat")

y_test1, x_test1 = svm_read_problem('ex4.dat')

# Train the model
# model = svm_train(y_train, x_train, '-h 0')
model = svm_train(y_test1, x_test1)

# Test the model
p_label, p_acc, p_val = svm_predict(y_test1, x_test1, model)


# Print the accuracy
# print(p_acc)

# 将字典形式的特征转换为数组形式
def convert_to_array(x_dict, num_features):
    x_array = np.zeros((len(x_dict), num_features))
    for i, row in enumerate(x_dict):
        for j in range(num_features):
            x_array[i, j] = row.get(j + 1, 0)
    return x_array


