import numpy as np
from math import floor, log

num_space = 16

def transform(number_list):
    maxl = -10000
    for i in range(num_space):
        minimum = min(number_list[i].flatten())
        nowl = floor(log(255 / (max(number_list[i].flatten()) - minimum), 2))
        if (nowl > maxl):
            maxl = nowl
    a = pow(2, maxl)
    result = []
    for i in range(num_space):
        minimum = min(number_list[i].flatten())
        temp = []
        for j in range(len(number_list[i])):
            temp.append(a * (number_list[i][j] - minimum))
        result.append(temp)
    return np.array(result, dtype='uint8')

table = np.load("lookup_table_c16.npy")
table_quan = transform(table)
np.save('lookup_table_c16_quan.npy', table_quan)