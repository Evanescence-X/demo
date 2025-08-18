import numpy as np
from numpy.ma.core import concatenate
'''
# 创建数组
# 创建一维数组
arr1 = np.array([1,2,3])
# 创建二维数组
arr2 = np.array([[1,2,3],[4,5,6]])

# 数组的属性
print("arr1 和 arr2 的形状是", str(arr1.shape) + str(arr2.shape))  # 形状
print("arr1 的元素总数是" , arr1.size)  # 数组的元素总数
print("arr2 的维度是" , arr2.ndim)  # 数组的维度数，也就是行
'''

'''
虽然 numpy 没有直接提供获取数组列数的属性，但是从其他属性可以得出
1、shape的输出本质是元组,有序的（type一下），所以可以对于shape的输出取后一列，即为列数
2、多维数组的本质是多个一维数组的组合，所以，取某一行的size，就是该多维数组的列数
'''
'''
print("arr2的列数是" , arr2.shape[1])
print("arr2的列数是" , arr2[0].size)
print("数组的数据类型", arr1.dtype)  # 数组的数据类型
print('每个元素的字节数' , arr2.itemsize)  # 每个元素的字节数

# 数组的操作

# 索引和切片
# 下标索引
arr3 = np.array([[1,2,3],[4,5,6],[7,8,9]])
print("第一行", arr3[0])
print("第二行第一个", arr3[1][0])

# 值索引
print("输出大于5的元素", arr3[arr3 > 5])

# 切片
print("前两行", arr3[:2])
print("第二列", arr3[:,1])
print("子数组", arr3[1:3,0:2])
np.expand_dims(arr3,axis=0)
print(arr3)


# 数组的拼接
brr1 = np.array([1,2,3])
brr2 = np.array([[1,2,3],[4,5,6]])
brr3 = np.array([[1,2,3],[4,5,6],[7,8,9]])
brr_extended = np.expand_dims(brr1,axis=0)
brr_extended2 = np.expand_dims(brr1,axis=1)
brr3_extended = np.expand_dims(brr3,axis=2)
print(brr1)
print(brr_extended)
print(brr1.shape)
print(brr_extended.shape)
print(brr_extended2.shape)
print(brr2.shape)
print(brr3_extended.shape)
print(brr3.shape)
'''

# 数组的拼接
arr1 = np.array([[1,2],[3,4]])
arr3 = np.array([[9,10]])
arr2 = np.array([[5,6],[7,8]])
arr_sum = np.concatenate((arr1,arr2),axis=1)
print(arr_sum)
print(np.hstack((arr1,arr2)))
print(np.vstack((arr1,arr2)))
''''''

# 数学运算
# 基本运算
arr1 = np.array([1,2,3])
arr2 = np.array([4,5,6])
print(arr1 - arr2)
print("点乘",np.dot(arr1,arr2))