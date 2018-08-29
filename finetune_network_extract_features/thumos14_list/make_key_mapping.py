'''
make Thumos14 category dictionary

Author: Lili Meng
Date: August 29th, 2018 
'''

import numpy as np

category_list = "category_list.txt"
lines = [line.strip() for line in open(category_list).readlines()]

category_dict = {}
for line in lines:
	key = int(line.split(' ')[0])-1
	print("key: ", key)
	value = line.split(' ')[1]
	print("value: ", value)
	category_dict[key] = value

print(category_dict)

np.save("category_dict.npy", category_dict)

	
