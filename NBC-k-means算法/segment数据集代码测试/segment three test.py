import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 连续型数据分类用正态分布公式
def getPro(theData, mean, var):
    a=1
    if (mean == 0) & (var == 0):
        return a
    else:
        pro=1 / (math.sqrt(2 * math.pi) * math.sqrt(var)) * math.exp(-(theData - mean) ** 2 / (2 * var))
        return pro

def getRandom(num):
    Ran = np.random.dirichlet(np.ones(num), size = 1)
    Ran = Ran.flatten()
    return Ran


'''
def CountP1(test):
    sum=1
    for i in range(0,60):
       sum*=getPro(test[i],
def CountP2(test):
    sum=1
    for i in  range(0,60):
        sum*=getPro(())
'''

X = np.loadtxt('[013]segment(0-1).txt')
# 其中有97
m = X.shape[1] - 1  # 属性数量
n = X.shape[0]  # 样本数目
T = 3
K = 7  # 类标记数量
# 主要过程：分组
# 去掉类标记
Class1 = 0
Class2 = 0
Class3 = 0
Class4 = 0
Class5 = 0
Class6 = 0
Class7 = 0

for i in X:
    if i[m] == 1:
        Class1 = Class1 + 1
    elif i[m] == 2:
        Class2 = Class2 + 1
    elif i[m] == 3:
        Class3 = Class3 + 1
    elif i[m] == 4:
        Class4 = Class4 + 1
    elif i[m] == 5:
        Class5 = Class5 + 1
    elif i[m] == 6:
        Class6 = Class6 + 1
    elif i[m] == 7:
        Class7 = Class7 + 1

train1 = int(Class1 * 0.5)
val1 = int(Class1 * 0.2)
test1 = Class1 - train1 - val1

train2 = int(Class2 * 0.5)
val2 = int(Class2 * 0.2)
test2 = Class2 - train2 - val2

train3 = int(Class3 * 0.5)
val3 = int(Class3 * 0.2)
test3 = Class3 - train3 - val3

train4 = int(Class4 * 0.5)
val4 = int(Class4 * 0.2)
test4 = Class4 - train4 - val4

train5 = int(Class5 * 0.5)
val5 = int(Class5 * 0.2)
test5 = Class5- train5 - val5

train6 = int(Class6 * 0.5)
val6 = int(Class6 * 0.2)
test6 = Class6 - train6 - val6

train7 = int(Class7 * 0.5)
val7 = int(Class7 * 0.2)
test7 = Class7 - train7 - val7

num2 = Class1 + Class2
num3 = Class1 + Class2 + Class3
num4 = num3 + Class4
num5 = num4 + Class5
num6 = num5 + Class6
num7 = num6 + Class7

# 随机产生多少个和为1的随机数W
G1 = [0, 1, 3, 4, 5, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18] # 16
G2 = [6, 8] # 2
G3 = [2] # 1

#随机训练集，验证集，测试集区

idx = np.random.choice(np.arange(Class1), size = train1, replace = False)
train_index1 = np.array(idx)
val_index1 = np.random.choice(np.delete(np.arange(Class1), train_index1), size = val1, replace = False)
test_index1 = np.delete(np.arange(Class1), np.append(train_index1, val_index1))

idx1 = np.random.choice(np.arange(Class2), size = train2, replace = False)
train_index2 = np.array(idx1)
val_index2 = np.random.choice(np.delete(np.arange(Class2), train_index2), size = val2, replace = False)
test_index2 = np.delete(np.arange(Class2), np.append(train_index2, val_index2))

idx2 = np.random.choice(np.arange(Class3), size = train3, replace = False)
train_index3 = np.array(idx2)
val_index3 = np.random.choice(np.delete(np.arange(Class3), train_index3), size = val3, replace = False)
test_index3 = np.delete(np.arange(Class3), np.append(train_index3, val_index3))

idx3 = np.random.choice(np.arange(Class4), size = train4, replace = False)
train_index4 = np.array(idx3)
val_index4 = np.random.choice(np.delete(np.arange(Class4), train_index4), size = val4, replace = False)
test_index4 = np.delete(np.arange(Class4), np.append(train_index4, val_index4))

idx4 = np.random.choice(np.arange(Class5), size = train5, replace = False)
train_index5 = np.array(idx4)
val_index5 = np.random.choice(np.delete(np.arange(Class5), train_index5), size = val5, replace = False)
test_index5 = np.delete(np.arange(Class5), np.append(train_index5, val_index5))

idx5 = np.random.choice(np.arange(Class6), size = train6, replace = False)
train_index6 = np.array(idx5)
val_index6 = np.random.choice(np.delete(np.arange(Class6), train_index6), size = val6, replace = False)
test_index6 = np.delete(np.arange(Class6), np.append(train_index6, val_index6))

idx6 = np.random.choice(np.arange(Class7), size = train7, replace = False)
train_index7 = np.array(idx6)
val_index7 = np.random.choice(np.delete(np.arange(Class7), train_index7), size = val7, replace = False)
test_index7 = np.delete(np.arange(Class7), np.append(train_index7, val_index7))

print("train_index1 =",list(train_index1))
print("val_index1 =",list(val_index1))
print("test_index1 =",list(test_index1))

print("train_index2 =",list(train_index2))
print("val_index2 =",list(val_index2))
print("test_index2 =",list(test_index2))

print("train_index3 =",list(train_index3))
print("val_index3 =",list(val_index3))
print("test_index3 =",list(test_index3))

print("train_index4 =",list(train_index4))
print("val_index4 =",list(val_index4))
print("test_index4 =",list(test_index4))

print("train_index5 =",list(train_index5))
print("val_index5 =",list(val_index5))
print("test_index5 =",list(test_index5))

print("train_index6 =",list(train_index6))
print("val_index6 =",list(val_index6))
print("test_index6 =",list(test_index6))

print("train_index7 =",list(train_index7))
print("val_index7 =",list(val_index7))
print("test_index7 =",list(test_index7))


#确认训练集，验证集，测试集区
train_index1 = [121, 63, 10, 285, 248, 163, 129, 305, 33, 93, 154, 292, 131, 275, 206, 315, 12, 7, 132, 270, 303, 151, 237, 2, 329, 158, 45, 6, 67, 286, 273, 284, 55, 249, 148, 293, 101, 117, 68, 326, 221, 89, 139, 17, 136, 177, 58, 126, 319, 165, 302, 38, 325, 192, 98, 157, 225, 298, 62, 110, 123, 25, 198, 244, 113, 167, 211, 259, 137, 21, 299, 35, 254, 140, 200, 181, 76, 120, 34, 272, 156, 173, 75, 164, 187, 143, 175, 236, 172, 133, 283, 51, 234, 212, 190, 239, 11, 279, 37, 1, 297, 87, 97, 18, 300, 169, 5, 321, 182, 82, 232, 202, 235, 288, 233, 247, 227, 201, 219, 317, 146, 114, 208, 73, 61, 166, 171, 314, 229, 269, 46, 322, 162, 24, 155, 125, 116, 238, 274, 218, 144, 149, 195, 59, 207, 138, 159, 142, 261, 91, 282, 85, 31, 52, 30, 80, 115, 86, 77, 310, 214, 152, 301, 262, 186]
val_index1 = [306, 199, 103, 78, 210, 258, 316, 96, 266, 150, 147, 111, 222, 153, 256, 23, 60, 40, 180, 213, 281, 4, 29, 74, 122, 105, 66, 118, 243, 0, 309, 79, 193, 84, 280, 47, 289, 205, 313, 217, 124, 179, 20, 119, 216, 81, 204, 260, 308, 19, 141, 128, 94, 184, 95, 203, 13, 253, 307, 268, 276, 99, 228, 88, 26, 22]
test_index1 = [3, 8, 9, 14, 15, 16, 27, 28, 32, 36, 39, 41, 42, 43, 44, 48, 49, 50, 53, 54, 56, 57, 64, 65, 69, 70, 71, 72, 83, 90, 92, 100, 102, 104, 106, 107, 108, 109, 112, 127, 130, 134, 135, 145, 160, 161, 168, 170, 174, 176, 178, 183, 185, 188, 189, 191, 194, 196, 197, 209, 215, 220, 223, 224, 226, 230, 231, 240, 241, 242, 245, 246, 250, 251, 252, 255, 257, 263, 264, 265, 267, 271, 277, 278, 287, 290, 291, 294, 295, 296, 304, 311, 312, 318, 320, 323, 324, 327, 328]
train_index2 = [256, 46, 48, 183, 22, 129, 323, 168, 271, 134, 25, 99, 58, 220, 228, 26, 297, 316, 5, 281, 84, 47, 210, 100, 133, 233, 117, 60, 16, 298, 200, 221, 3, 272, 246, 307, 172, 222, 125, 306, 216, 196, 59, 178, 191, 258, 7, 113, 318, 291, 126, 36, 154, 310, 76, 145, 305, 17, 66, 315, 193, 283, 234, 287, 68, 91, 293, 34, 286, 45, 185, 118, 211, 295, 265, 28, 264, 166, 49, 92, 245, 229, 71, 81, 274, 93, 2, 320, 55, 300, 198, 87, 95, 278, 236, 240, 176, 97, 226, 128, 214, 32, 114, 165, 89, 314, 251, 27, 161, 77, 98, 83, 155, 289, 243, 8, 141, 127, 241, 194, 277, 171, 212, 90, 149, 308, 50, 162, 239, 270, 116, 101, 135, 9, 123, 303, 189, 121, 268, 253, 327, 218, 38, 182, 257, 302, 23, 235, 309, 279, 186, 63, 11, 263, 75, 144, 217, 321, 199, 62, 180, 33, 151, 215, 254]
val_index2 = [294, 276, 187, 275, 296, 86, 54, 85, 164, 73, 13, 324, 225, 224, 209, 280, 69, 261, 20, 304, 273, 24, 285, 205, 110, 107, 148, 203, 112, 190, 120, 157, 250, 39, 147, 213, 322, 40, 328, 177, 262, 248, 282, 136, 29, 14, 158, 174, 119, 53, 102, 79, 266, 108, 21, 82, 184, 227, 169, 195, 74, 42, 170, 179, 319, 140]
test_index2 = [0, 1, 4, 6, 10, 12, 15, 18, 19, 30, 31, 35, 37, 41, 43, 44, 51, 52, 56, 57, 61, 64, 65, 67, 70, 72, 78, 80, 88, 94, 96, 103, 104, 105, 106, 109, 111, 115, 122, 124, 130, 131, 132, 137, 138, 139, 142, 143, 146, 150, 152, 153, 156, 159, 160, 163, 167, 173, 175, 181, 188, 192, 197, 201, 202, 204, 206, 207, 208, 219, 223, 230, 231, 232, 237, 238, 242, 244, 247, 249, 252, 255, 259, 260, 267, 269, 284, 288, 290, 292, 299, 301, 311, 312, 313, 317, 325, 326, 329]
train_index3 = [136, 234, 243, 34, 281, 185, 186, 111, 300, 96, 73, 122, 269, 91, 134, 46, 58, 222, 127, 150, 52, 157, 265, 104, 64, 302, 156, 154, 327, 197, 170, 53, 28, 0, 94, 311, 47, 118, 254, 49, 31, 93, 87, 214, 219, 116, 121, 103, 231, 299, 233, 287, 216, 75, 169, 172, 225, 33, 13, 190, 213, 252, 307, 163, 220, 288, 244, 153, 54, 325, 27, 251, 303, 144, 177, 39, 3, 180, 23, 193, 313, 246, 205, 141, 112, 278, 261, 167, 264, 101, 279, 26, 202, 181, 174, 208, 191, 289, 60, 82, 70, 204, 132, 140, 148, 308, 158, 319, 282, 198, 15, 130, 229, 7, 206, 66, 305, 108, 5, 292, 179, 100, 207, 76, 1, 274, 192, 253, 32, 138, 275, 189, 248, 142, 315, 84, 312, 89, 41, 123, 314, 255, 71, 196, 24, 97, 267, 22, 309, 291, 316, 85, 86, 35, 160, 223, 165, 182, 117, 318, 290, 260, 68, 45, 95]
val_index3 = [57, 249, 164, 37, 227, 178, 212, 149, 120, 239, 25, 88, 62, 236, 321, 106, 128, 232, 139, 131, 72, 78, 83, 147, 173, 211, 256, 155, 263, 102, 80, 250, 238, 56, 320, 98, 326, 237, 221, 92, 44, 168, 38, 226, 135, 159, 42, 210, 90, 286, 48, 17, 51, 184, 195, 99, 262, 129, 268, 18, 194, 69, 124, 224, 273, 126]
test_index3 = [2, 4, 6, 8, 9, 10, 11, 12, 14, 16, 19, 20, 21, 29, 30, 36, 40, 43, 50, 55, 59, 61, 63, 65, 67, 74, 77, 79, 81, 105, 107, 109, 110, 113, 114, 115, 119, 125, 133, 137, 143, 145, 146, 151, 152, 161, 162, 166, 171, 175, 176, 183, 187, 188, 199, 200, 201, 203, 209, 215, 217, 218, 228, 230, 235, 240, 241, 242, 245, 247, 257, 258, 259, 266, 270, 271, 272, 276, 277, 280, 283, 284, 285, 293, 294, 295, 296, 297, 298, 301, 304, 306, 310, 317, 322, 323, 324, 328, 329]
train_index4 = [138, 30, 314, 85, 168, 270, 208, 56, 312, 2, 143, 264, 214, 75, 200, 329, 16, 310, 136, 256, 118, 4, 135, 165, 234, 301, 73, 115, 49, 309, 66, 28, 146, 292, 78, 253, 221, 290, 262, 46, 1, 247, 21, 274, 54, 202, 62, 0, 315, 191, 181, 257, 207, 97, 199, 179, 91, 267, 194, 232, 113, 251, 231, 31, 147, 63, 246, 111, 95, 51, 174, 37, 59, 120, 108, 237, 141, 277, 33, 81, 133, 169, 5, 67, 9, 11, 197, 266, 259, 88, 196, 212, 261, 205, 43, 243, 328, 103, 291, 195, 209, 101, 122, 140, 77, 34, 154, 52, 126, 29, 254, 76, 10, 132, 245, 164, 316, 148, 233, 272, 123, 224, 218, 204, 210, 311, 13, 137, 50, 12, 325, 58, 252, 45, 79, 121, 323, 306, 230, 158, 39, 170, 142, 320, 155, 275, 57, 26, 227, 171, 106, 35, 297, 72, 70, 268, 131, 53, 244, 222, 308, 144, 36, 130, 48]
val_index4 = [236, 61, 250, 217, 283, 8, 258, 102, 127, 40, 125, 116, 99, 276, 166, 296, 187, 185, 269, 145, 3, 201, 64, 55, 157, 278, 279, 327, 110, 203, 281, 117, 305, 41, 225, 44, 293, 271, 22, 303, 260, 74, 20, 96, 87, 160, 180, 307, 172, 265, 192, 84, 273, 149, 211, 42, 242, 47, 220, 69, 182, 228, 313, 255, 92, 321]
test_index4 = [6, 7, 14, 15, 17, 18, 19, 23, 24, 25, 27, 32, 38, 60, 65, 68, 71, 80, 82, 83, 86, 89, 90, 93, 94, 98, 100, 104, 105, 107, 109, 112, 114, 119, 124, 128, 129, 134, 139, 150, 151, 152, 153, 156, 159, 161, 162, 163, 167, 173, 175, 176, 177, 178, 183, 184, 186, 188, 189, 190, 193, 198, 206, 213, 215, 216, 219, 223, 226, 229, 235, 238, 239, 240, 241, 248, 249, 263, 280, 282, 284, 285, 286, 287, 288, 289, 294, 295, 298, 299, 300, 302, 304, 317, 318, 319, 322, 324, 326]
train_index5 = [220, 189, 173, 305, 148, 126, 241, 6, 122, 191, 188, 214, 195, 52, 179, 166, 144, 259, 253, 322, 150, 227, 303, 22, 245, 219, 260, 270, 0, 110, 29, 86, 36, 239, 237, 128, 19, 194, 328, 48, 82, 218, 178, 294, 246, 290, 201, 182, 292, 88, 30, 156, 125, 326, 221, 17, 211, 170, 278, 298, 103, 45, 327, 32, 97, 55, 87, 158, 249, 165, 186, 39, 171, 143, 162, 289, 208, 69, 317, 31, 274, 134, 8, 60, 79, 311, 136, 83, 172, 41, 168, 47, 176, 80, 291, 37, 196, 99, 123, 243, 273, 130, 226, 27, 3, 316, 297, 26, 142, 66, 209, 108, 46, 250, 210, 100, 314, 72, 145, 213, 104, 267, 167, 105, 131, 312, 256, 90, 200, 197, 279, 264, 149, 114, 68, 285, 169, 257, 277, 181, 33, 324, 281, 24, 43, 276, 247, 91, 5, 18, 184, 321, 115, 151, 129, 118, 140, 157, 232, 229, 180, 310, 101, 154, 240]
val_index5 = [198, 102, 2, 9, 320, 106, 85, 299, 304, 38, 53, 309, 252, 34, 107, 163, 242, 133, 49, 89, 141, 215, 92, 293, 206, 235, 42, 138, 204, 44, 137, 262, 234, 59, 11, 202, 25, 14, 248, 74, 147, 119, 120, 193, 296, 301, 315, 302, 251, 307, 187, 54, 94, 84, 185, 280, 23, 300, 261, 58, 199, 238, 159, 174, 212, 236]
test_index5 = [1, 4, 7, 10, 12, 13, 15, 16, 20, 21, 28, 35, 40, 50, 51, 56, 57, 61, 62, 63, 64, 65, 67, 70, 71, 73, 75, 76, 77, 78, 81, 93, 95, 96, 98, 109, 111, 112, 113, 116, 117, 121, 124, 127, 132, 135, 139, 146, 152, 153, 155, 160, 161, 164, 175, 177, 183, 190, 192, 203, 205, 207, 216, 217, 222, 223, 224, 225, 228, 230, 231, 233, 244, 254, 255, 258, 263, 265, 266, 268, 269, 271, 272, 275, 282, 283, 284, 286, 287, 288, 295, 306, 308, 313, 318, 319, 323, 325, 329]
train_index6 = [102, 246, 213, 272, 180, 154, 240, 178, 112, 295, 66, 268, 59, 273, 301, 116, 289, 87, 238, 311, 256, 236, 220, 6, 31, 18, 196, 181, 319, 105, 269, 260, 307, 28, 158, 93, 128, 58, 245, 161, 285, 227, 111, 314, 95, 271, 115, 145, 11, 199, 131, 84, 201, 120, 133, 142, 53, 292, 322, 114, 104, 94, 207, 4, 182, 39, 42, 109, 318, 276, 279, 52, 129, 303, 46, 38, 194, 140, 73, 250, 15, 174, 155, 110, 254, 221, 249, 103, 321, 183, 204, 123, 67, 8, 86, 14, 144, 16, 286, 263, 41, 29, 317, 197, 329, 0, 62, 96, 291, 7, 278, 169, 168, 323, 43, 40, 77, 239, 172, 92, 64, 195, 159, 176, 151, 65, 294, 134, 210, 225, 24, 327, 98, 179, 121, 175, 253, 13, 167, 184, 135, 88, 49, 79, 125, 50, 173, 243, 36, 100, 205, 216, 230, 78, 51, 187, 148, 320, 309, 82, 1, 251, 60, 113, 163]
val_index6 = [157, 298, 232, 150, 68, 138, 63, 35, 226, 304, 107, 315, 257, 149, 177, 224, 212, 211, 252, 283, 156, 37, 288, 265, 164, 214, 5, 57, 76, 242, 45, 124, 153, 55, 310, 71, 72, 302, 27, 297, 25, 262, 198, 126, 266, 228, 34, 9, 26, 33, 139, 90, 305, 69, 30, 287, 281, 188, 47, 160, 267, 234, 74, 259, 258, 162]
test_index6 = [2, 3, 10, 12, 17, 19, 20, 21, 22, 23, 32, 44, 48, 54, 56, 61, 70, 75, 80, 81, 83, 85, 89, 91, 97, 99, 101, 106, 108, 117, 118, 119, 122, 127, 130, 132, 136, 137, 141, 143, 146, 147, 152, 165, 166, 170, 171, 185, 186, 189, 190, 191, 192, 193, 200, 202, 203, 206, 208, 209, 215, 217, 218, 219, 222, 223, 229, 231, 233, 235, 237, 241, 244, 247, 248, 255, 261, 264, 270, 274, 275, 277, 280, 282, 284, 290, 293, 296, 299, 300, 306, 308, 312, 313, 316, 324, 325, 326, 328]
train_index7 = [3, 82, 179, 315, 262, 268, 63, 275, 195, 114, 302, 217, 224, 144, 176, 325, 215, 205, 132, 218, 203, 33, 251, 199, 85, 289, 216, 245, 255, 204, 261, 81, 75, 41, 2, 89, 272, 167, 141, 83, 147, 129, 207, 28, 111, 29, 270, 124, 40, 119, 178, 219, 80, 59, 274, 108, 196, 47, 200, 294, 35, 233, 260, 98, 143, 64, 308, 234, 155, 281, 27, 122, 46, 212, 103, 267, 183, 319, 248, 312, 96, 139, 131, 208, 140, 256, 125, 258, 25, 324, 276, 95, 151, 213, 106, 133, 271, 61, 314, 184, 285, 50, 74, 280, 88, 202, 232, 153, 249, 116, 193, 31, 138, 305, 328, 206, 24, 225, 72, 137, 146, 145, 128, 109, 110, 13, 221, 79, 166, 76, 320, 57, 316, 326, 223, 113, 321, 112, 84, 4, 120, 186, 317, 0, 20, 214, 15, 107, 77, 18, 174, 283, 310, 7, 73, 78, 187, 62, 301, 157, 45, 209, 156, 134, 295]
val_index7 = [254, 198, 126, 26, 190, 181, 53, 11, 244, 307, 117, 123, 172, 327, 210, 8, 264, 273, 32, 6, 127, 313, 105, 135, 161, 21, 70, 37, 189, 259, 43, 265, 104, 177, 52, 171, 93, 17, 277, 9, 287, 5, 130, 292, 236, 227, 180, 38, 175, 49, 1, 86, 90, 229, 279, 293, 100, 311, 115, 299, 329, 235, 66, 297, 169, 23]
test_index7 = [10, 12, 14, 16, 19, 22, 30, 34, 36, 39, 42, 44, 48, 51, 54, 55, 56, 58, 60, 65, 67, 68, 69, 71, 87, 91, 92, 94, 97, 99, 101, 102, 118, 121, 136, 142, 148, 149, 150, 152, 154, 158, 159, 160, 162, 163, 164, 165, 168, 170, 173, 182, 185, 188, 191, 192, 194, 197, 201, 211, 220, 222, 226, 228, 230, 231, 237, 238, 239, 240, 241, 242, 243, 246, 247, 250, 252, 253, 257, 263, 266, 269, 278, 282, 284, 286, 288, 290, 291, 296, 298, 300, 303, 304, 306, 309, 318, 322, 323]




W = getRandom(m * K) * 100
W=[6310868.18372201, 8150730.410814098, 3278038.7912354977, 4601356.423864662, 6138969.192532067, 4808030.23217565, 7816049.329185358, 4017979.2893114127, 4879792.141455387, 7068540.63231012, 7117794.742798939, 2987480.5737142977, 6488535.8210450215, 2294591.5891158422, 4275760.326288266, 4543072.223327982, 5701483.877526386, 7405005.967307294, 2004368.0035858145, 6851224.443306722, 2560730.2809687247, 5284258.592418973, 4471533.421989102, 5616382.809077669, 7811880.464286293, 4579915.8264941955, 5020644.962631001, 3529040.099346998, 5675655.133116107, 6465604.221047359, 3369850.5638328483, 7488840.570935939, 6727733.241114889, 6686768.525883501, 3437433.0444020364, 6594046.141846614, 4828874.283928563, 6064425.828958011, 3332796.1459663836, 3708711.788684724, 5745636.902062194, 7312498.590039299, 4839808.849551124, 3691347.138951203, 3020788.4688295457, 6296110.420820633, 2887512.189131877, 4176297.738299408, 2982117.87746777, 3681575.1498897932, 6750739.590296398, 5601529.044631539, 4234247.79216665, 2261641.4864294874, 3948809.8370855832, 6849700.432942453, 8147759.664469281, 6588144.982730821, 4935903.009628595, 1092630.4546584643, 6619920.409172261, 7474277.101605672, 4146750.6903849607, 5224093.29975173, 5530568.184065126, 7157736.609989081, 5368359.543035116, 8856887.521323593, 4782974.15788219, 5270008.738328747, 6619744.0731848525, 3092053.0251276465, 5015724.829249694, 4714315.824611786, 4724972.1467456035, 1820137.910482433, 2786369.0476645343, 3813241.6000671145, 5060818.303106255, 6595174.377716644, 5399506.8107075505, 6745117.166061372, 1098417.2899861669, 5699220.518570676, 4605525.629872149, 2357942.389167116, 6827898.428407558, 4984427.174462675, 4522816.057431479, 5361182.224171429, 5352497.398205334, 5684711.21800661, 2781905.0284125013, 6449992.244024576, 3330874.7923361305, 6867585.580706671, 3669225.1369962934, 5999040.052273722, 3660908.4647311177, 7575414.549588914, 3723862.0383124305, 4437434.315275338, 8178997.170641647, 5622954.306207998, 7491849.586249797, 3264632.394549339, 6086679.081482353, 5203345.921114776, 5955669.047258694, 5999282.294237204, 5585225.501532739, 4327747.49947671, 2546351.7727373345, 3488746.2291679615, 6653417.364661736, 5147942.312195318, 8984818.416221514, 4243537.81198522, 5976274.241541719, 3856193.4824742493, 6457525.894279819, 4553884.716722455, 6737650.609366665, 4530508.707341811, 5674902.987480316, 3475369.3501521763, 6841363.032471495, 6932991.312800365, 3245749.336013372, 6968709.696074532, 6000514.0670648385, 1824197.956655119, 4535756.3349121]


# 求类1的分组情况
NewArray1 = np.ones((Class1, T + 1))
# 第0组
W1 = W[0:16]
for i in range(0, Class1):
    add1 = 0
    for j in range(0, 16):
        add1 += W1[j] * X[i, G1[j]]
    NewArray1[i][0] = add1
# 第1组
W2 = W[16:18]
for i in range(0, Class1):
    add2 = 0
    for j in range(0, 2):
        add2 += W2[j] * X[i, G2[j]]
    NewArray1[i][1] = add2
# 第2组
W3 = W[18:19]
for i in range(0, Class1):
    add3 = 0
    for j in range(0, 1):
        add3 += W3[j] * X[i, G3[j]]
    NewArray1[i][2] = add3
# print(NewArray1)

# 求类2的分组情况
NewArray2 = np.ones((Class2, T + 1)) * 2
# 第0组
W4 = W[19:35]
for i in range(Class1, Class1 + Class2):
    add1 = 0
    for j in range(0, 16):
        add1 += W4[j] * X[i, G1[j]]
    NewArray2[i - Class1][0] = add1
# 第1组
W5 = W[35:37]
for i in range(Class1, Class1 + Class2):
    add2 = 0
    for j in range(0, 2):
        add2 += W5[j] * X[i, G2[j]]
    NewArray2[i - Class1][1] = add2
# 第2组
W6 = W[37:38]
for i in range(Class1, Class1 + Class2):
    add3 = 0
    for j in range(0, 1):
        add3 += W6[j] * X[i, G3[j]]
    NewArray2[i - Class1][2] = add3
# print(NewArray2)

# 求类3的分组情况
NewArray3 = np.ones((Class3, T + 1)) * 3
# 第0组
W7 = W[38:54]
for i in range(Class1 + Class2, Class1 + Class2 + Class3):
    add1 = 0
    for j in range(0, 16):
        add1 += W7[j] * X[i, G1[j]]
    NewArray3[i - Class1 - Class2][0] = add1
# 第1组
W8 = W[54:56]
for i in range(Class1 + Class2, Class1 + Class2 + Class3):
    add2 = 0
    for j in range(0, 2):
        add2 += W8[j] * X[i, G2[j]]
    NewArray3[i - Class1 - Class2][1] = add2
# 第2组
W9 = W[56:57]
for i in range(Class1 + Class2, Class1 + Class2 + Class3):
    add3 = 0
    for j in range(0, 1):
        add3 += W9[j] * X[i, G3[j]]
    NewArray3[i - Class1 - Class2][2] = add3
# print(NewArray3)

# 求类4的分组情况
NewArray4 = np.ones((Class4, T + 1)) * 4
# 第0组
W10 = W[57:73]
for i in range(Class1 + Class2 + Class3, Class1 + Class2 + Class3 + Class4):
    add1 = 0
    for j in range(0, 16):
        add1 += W10[j] * X[i, G1[j]]
    NewArray4[i - Class1 - Class2 - Class3][0] = add1
# 第1组
W11 = W[73:75]
for i in range(Class1 + Class2 + Class3, Class1 + Class2 + Class3 + Class4):
    add2 = 0
    for j in range(0, 2):
        add2 += W11[j] * X[i, G2[j]]
    NewArray4[i - Class1 - Class2 - Class3][1] = add2
# 第2组
W12 = W[75:76]
for i in range(Class1 + Class2 + Class3, Class1 + Class2 + Class3 + Class4):
    add3 = 0
    for j in range(0, 1):
        add3 += W12[j] * X[i, G3[j]]
    NewArray4[i - Class1 - Class2 - Class3][2] = add3
# print(NewArray4)

# 求类5的分组情况
NewArray5 = np.ones((Class5, T + 1)) * 5
# 第0组
W13 = W[76:92]
for i in range(Class1 + Class2 + Class3 + Class4, Class1 + Class2 + Class3 + Class4 + Class5):
    add1 = 0
    for j in range(0, 16):
        add1 += W13[j] * X[i, G1[j]]
    NewArray5[i - Class1 - Class2 - Class3 - Class4][0] = add1
# 第1组
W14 = W[92:94]
for i in range(Class1 + Class2 + Class3 + Class4, Class1 + Class2 + Class3 + Class4 + Class5):
    add2 = 0
    for j in range(0, 2):
        add2 += W14[j] * X[i, G2[j]]
    NewArray5[i - Class1 - Class2 - Class3 - Class4][1] = add2
# 第2组
W15 = W[94:95]
for i in range(Class1 + Class2 + Class3 + Class4, Class1 + Class2 + Class3 + Class4 + Class5):
    add3 = 0
    for j in range(0, 1):
        add3 += W15[j] * X[i, G3[j]]
    NewArray5[i - Class1 - Class2 - Class3 - Class4][2] = add3
# print(NewArray5)

# 求类6的分组情况
NewArray6 = np.ones((Class6, T + 1)) * 6
# 第0组
W16 = W[95:111]
for i in range(Class1 + Class2 + Class3 + Class4 + Class5, Class1 + Class2 + Class3 + Class4 + Class5 + Class6):
    add1 = 0
    for j in range(0, 16):
        add1 += W16[j] * X[i, G1[j]]
    NewArray6[i - Class1 - Class2 - Class3 - Class4 - Class5][0] = add1
# 第1组
W17 = W[111:113]
for i in range(Class1 + Class2 + Class3 + Class4 + Class5, Class1 + Class2 + Class3 + Class4 + Class5 + Class6):
    add2 = 0
    for j in range(0, 2):
        add2 += W17[j] * X[i, G2[j]]
    NewArray6[i - Class1 - Class2 - Class3 - Class4 - Class5][1] = add2
# 第2组
W18 = W[113:114]
for i in range(Class1 + Class2 + Class3 + Class4 + Class5, Class1 + Class2 + Class3 + Class4 + Class5 + Class6):
    add3 = 0
    for j in range(0, 1):
        add3 += W18[j] * X[i, G3[j]]
    NewArray6[i - Class1 - Class2 - Class3 - Class4 - Class5][2] = add3

# 求类7的分组情况
NewArray7 = np.ones((Class6, T + 1)) * 7
# 第0组
W19 = W[114:130]
for i in range(num6, num7):
    add1 = 0
    for j in range(0, 16):
        add1 += W19[j] * X[i, G1[j]]
    NewArray7[i - num6][0] = add1
# 第1组
W20 = W[130:132]
for i in range(num6, num7):
    add2 = 0
    for j in range(0, 2):
        add2 += W20[j] * X[i, G2[j]]
    NewArray7[i - num6][1] = add2
# 第2组
W21 = W[132:133]
for i in range(num6, num7):
    add3 = 0
    for j in range(0, 1):
        add3 += W21[j] * X[i, G3[j]]
    NewArray7[i - num6][2] = add3
# print(NewArray1)
# 合并两个数组，得到真正的合并数据结果
NewArray = np.vstack((NewArray1,NewArray2,NewArray3,NewArray4,NewArray5,NewArray6,NewArray7))
NewArray1 = np.delete(NewArray, T, axis = 1)
df=pd.DataFrame(NewArray1)
sns.pairplot(df)
plt.show()
# print(NewArray)
X=NewArray
# 随机抽取样本训练集和测试集样本
print(X)
X1 = X[0:Class1, :]
X2 = X[Class1:num2, :]
X3 = X[num2:num3, :]
X4 = X[num3:num4, :]
# print(X4)
X5 = X[num4:num5, :]
# print(X5)
X6 = X[num5:num6, :]
X7 = X[num6:num7, :]

Data1 = X1[train_index1, :]
Data2 = X2[train_index2, :]
Data3 = X3[train_index3, :]
Data4 = X4[train_index4, :]
Data5 = X5[train_index5, :]
Data6 = X6[train_index6, :]
Data7 = X7[train_index7, :]

testSet1 = np.delete(X1[test_index1, :], T, axis = 1)
testSet2 = np.delete(X2[test_index2, :], T, axis = 1)
testSet3 = np.delete(X3[test_index3, :], T, axis = 1)
testSet4 = np.delete(X4[test_index4, :], T, axis = 1)
testSet5 = np.delete(X5[test_index5, :], T, axis = 1)
testSet6 = np.delete(X6[test_index6, :], T, axis = 1)
testSet7 = np.delete(X7[test_index7, :], T, axis = 1)

trainSet1 = np.delete(Data1, T, axis = 1)
trainSet2 = np.delete(Data2, T, axis = 1)
trainSet3 = np.delete(Data3, T, axis = 1)
trainSet4 = np.delete(Data4, T, axis = 1)
trainSet5 = np.delete(Data5, T, axis = 1)
trainSet6 = np.delete(Data6, T, axis = 1)
trainSet7 = np.delete(Data7, T, axis = 1)

valSet1=np.delete(X1[val_index1,:],T,axis = 1)
valSet2=np.delete(X2[val_index2,:],T,axis = 1)
valSet3=np.delete(X3[val_index3,:],T,axis = 1)
valSet4=np.delete(X4[val_index4,:],T,axis = 1)
valSet5=np.delete(X5[val_index5,:],T,axis = 1)
valSet6=np.delete(X6[val_index6,:],T,axis = 1)
valSet7=np.delete(X7[val_index7,:],T,axis = 1)

# 求各类对应属性的均值和方差
Mean1 = np.mean(trainSet1, axis = 0)
Mean2 = np.mean(trainSet2, axis = 0)
Mean3 = np.mean(trainSet3, axis = 0)
Mean4 = np.mean(trainSet4, axis = 0)
Mean5 = np.mean(trainSet5, axis = 0)
Mean6 = np.mean(trainSet6, axis = 0)
Mean7 = np.mean(trainSet7, axis = 0)

var1 = np.mean(trainSet1, axis = 0)
var2 = np.mean(trainSet2, axis = 0)
var3 = np.mean(trainSet3, axis = 0)
var4 = np.mean(trainSet4, axis = 0)
var5 = np.mean(trainSet5, axis = 0)
var6 = np.mean(trainSet6, axis = 0)
var7 = np.mean(trainSet7, axis = 0)
alltrain = train1 + train2 + train3 + train4 + train5 + train6 + train7
# 先求P(C)
Pro1 = (train1 + 1) / (alltrain + 7)
Pro2 = (train2 + 1) / (alltrain + 7)
Pro3 = (train3 + 1) / (alltrain + 7)
Pro4 = (train4 + 1) / (alltrain + 7)
Pro5 = (train5 + 1) / (alltrain + 7)
Pro6 = (train6 + 1) / (alltrain + 7)
Pro7 = (train7 + 1) / (alltrain + 7)

add = 0
for i in range(0, test1):
    sum = 1
    for j in range(0, T):
        sum *= getPro(testSet1[i][j], Mean1[j], var1[j])
    sum1 = 1
    for j in range(0, T):
        sum1 *= getPro(testSet1[i][j], Mean2[j], var2[j])
    sum2 = 1
    for j in range(0, T):
        sum2 *= getPro(testSet1[i][j], Mean3[j], var3[j])
    sum3 = 1
    for j in range(0, T):
        sum3 *= getPro(testSet1[i][j], Mean4[j], var4[j])
    sum4 = 1
    for j in range(0, T):
        sum4 *= getPro(testSet1[i][j], Mean5[j], var5[j])
    sum5 = 1
    for j in range(0, T):
        sum5 *= getPro(testSet1[i][j], Mean6[j], var6[j])
    sum6 = 1
    for j in range(0, T):
        sum6 *= getPro(testSet1[i][j], Mean7[j], var7[j])
    if (Pro1 * sum >= Pro2 * sum1) & (Pro1 * sum >= Pro3 * sum2) & (Pro1 * sum >= Pro4 * sum3) & (
            Pro1 * sum >= Pro5 * sum4) & (Pro1 * sum >= Pro6 * sum5) & (Pro1 * sum >= Pro7 * sum6):
        add += 1
    else:
        add += 0
print("第一类正确数量(总数27)：")
print(add)
add1 = 0
for i in range(0, test2):
    sum = 1
    for j in range(0, T):
        sum *= getPro(testSet2[i][j], Mean1[j], var1[j])
    sum1 = 1
    for j in range(0, T):
        sum1 *= getPro(testSet2[i][j], Mean2[j], var2[j])
    sum2 = 1
    for j in range(0, T):
        sum2 *= getPro(testSet2[i][j], Mean3[j], var3[j])
    sum3 = 1
    for j in range(0, T):
        sum3 *= getPro(testSet2[i][j], Mean4[j], var4[j])
    sum4 = 1
    for j in range(0, T):
        sum4 *= getPro(testSet2[i][j], Mean5[j], var5[j])
    sum5 = 1
    for j in range(0, T):
        sum5 *= getPro(testSet2[i][j], Mean6[j], var6[j])
    sum6 = 1
    for j in range(0, T):
        sum6 *= getPro(testSet2[i][j], Mean7[j], var7[j])
    if (Pro2 * sum1 >= Pro1 * sum) & (Pro2 * sum1 >= Pro3 * sum2) & (Pro2 * sum1 >= Pro4 * sum3) & (
            Pro2 * sum1 >= Pro5 * sum4) & (Pro2 * sum1 >= Pro6 * sum5) & (Pro2 * sum1 >= Pro7 * sum6):
        add1 += 1
    else:
        add1 += 0
print("第二类正确数量(总数34)：")
print(add1)

# 计算第三类
add2 = 0
for i in range(0, test3):
    sum = 1
    for j in range(0, T):
        sum *= getPro(testSet3[i][j], Mean1[j], var1[j])
    sum1 = 1
    for j in range(0, T):
        sum1 *= getPro(testSet3[i][j], Mean2[j], var2[j])
    sum2 = 1
    for j in range(0, T):
        sum2 *= getPro(testSet3[i][j], Mean3[j], var3[j])
    sum3 = 1
    for j in range(0, T):
        sum3 *= getPro(testSet3[i][j], Mean4[j], var4[j])
    sum4 = 1
    for j in range(0, T):
        sum4 *= getPro(testSet3[i][j], Mean5[j], var5[j])
    sum5 = 1
    for j in range(0, T):
        sum5 *= getPro(testSet3[i][j], Mean6[j], var6[j])
    sum6 = 1
    for j in range(0, T):
        sum6 *= getPro(testSet3[i][j], Mean7[j], var7[j])
    if (Pro3 * sum2 >= Pro1 * sum) & (Pro3 * sum2 >= Pro2 * sum1) & (Pro3 * sum2 >= Pro4 * sum3) & (
            Pro3 * sum2 >= Pro5 * sum4) & (Pro3 * sum2 >= Pro6 * sum5) & (Pro3 * sum2 >= Pro7 * sum6):
        add2 += 1
    else:
        add2 += 0
print("第三类正确数量(总数204)：")
print(add2)

add3 = 0
for i in range(0, test4):
    sum = 1
    for j in range(0, T):
        sum *= getPro(testSet4[i][j], Mean1[j], var1[j])
    sum1 = 1
    for j in range(0, T):
        sum1 *= getPro(testSet4[i][j], Mean2[j], var2[j])
    sum2 = 1
    for j in range(0, T):
        sum2 *= getPro(testSet4[i][j], Mean3[j], var3[j])
    sum3 = 1
    for j in range(0, T):
        sum3 *= getPro(testSet4[i][j], Mean4[j], var4[j])
    sum4 = 1
    for j in range(0, T):
        sum4 *= getPro(testSet4[i][j], Mean5[j], var5[j])
    sum5 = 1
    for j in range(0, T):
        sum5 *= getPro(testSet4[i][j], Mean6[j], var6[j])
    sum6 = 1
    for j in range(0, T):
        sum6 *= getPro(testSet4[i][j], Mean7[j], var7[j])
    if (Pro4 * sum3 >= Pro1 * sum) & (Pro4 * sum3 >= Pro2 * sum1) & (Pro4 * sum3 >= Pro3 * sum2) & (
            Pro4 * sum3 >= Pro5 * sum4) & (Pro4 * sum3 >= Pro6 * sum5) & (Pro4 * sum3 >= Pro7 * sum6):
        add3 += 1
    else:
        add3 += 0
print("第四类正确数量(总数192)：")
print(add3)

add4 = 0
for i in range(0, test5):
    sum = 1
    for j in range(0, T):
        sum *= getPro(testSet5[i][j], Mean1[j], var1[j])
    sum1 = 1
    for j in range(0, T):
        sum1 *= getPro(testSet5[i][j], Mean2[j], var2[j])
    sum2 = 1
    for j in range(0, T):
        sum2 *= getPro(testSet5[i][j], Mean3[j], var3[j])
    sum3 = 1
    for j in range(0, T):
        sum3 *= getPro(testSet5[i][j], Mean4[j], var4[j])
    sum4 = 1
    for j in range(0, T):
        sum4 *= getPro(testSet5[i][j], Mean5[j], var5[j])
    sum5 = 1
    for j in range(0, T):
        sum5 *= getPro(testSet5[i][j], Mean6[j], var6[j])
    sum6 = 1
    for j in range(0, T):
        sum6 *= getPro(testSet5[i][j], Mean7[j], var7[j])
    if (Pro5 * sum4 >= Pro1 * sum) & (Pro5 * sum4 >= Pro2 * sum1) & (Pro5 * sum4 >= Pro3 * sum2) & (
            Pro5 * sum4 >= Pro4 * sum3) & (Pro5 * sum4 >= Pro6 * sum5) & (Pro5 * sum4 >= Pro7 * sum6):
        add4 += 1
    else:
        add4 += 0
print("第五类正确数量(总数60)：")
print(add4)

add5 = 0
for i in range(0, test6):
    sum = 1
    for j in range(0, T):
        sum *= getPro(testSet6[i][j], Mean1[j], var1[j])
    sum1 = 1
    for j in range(0, T):
        sum1 *= getPro(testSet6[i][j], Mean2[j], var2[j])
    sum2 = 1
    for j in range(0, T):
        sum2 *= getPro(testSet6[i][j], Mean3[j], var3[j])
    sum3 = 1
    for j in range(0, T):
        sum3 *= getPro(testSet6[i][j], Mean4[j], var4[j])
    sum4 = 1
    for j in range(0, T):
        sum4 *= getPro(testSet6[i][j], Mean5[j], var5[j])
    sum5 = 1
    for j in range(0, T):
        sum5 *= getPro(testSet6[i][j], Mean6[j], var6[j])
    sum6 = 1
    for j in range(0, T):
        sum6 *= getPro(testSet6[i][j], Mean7[j], var7[j])
    if (Pro6 * sum5 >= Pro1 * sum) & (Pro6 * sum5 >= Pro2 * sum1) & (Pro6 * sum5 >= Pro3 * sum2) & (
            Pro6 * sum5 >= Pro4 * sum3) & (Pro6 * sum5 >= Pro5 * sum4) & (Pro6 * sum5 >= Pro7 * sum6):
        add5 += 1
    else:
        add5 += 0
print("第六类正确数量(总数6)：")
print(add5)
add6 = 0
for i in range(0, test7):
    sum = 1
    for j in range(0, T):
        sum *= getPro(testSet7[i][j], Mean1[j], var1[j])
    sum1 = 1
    for j in range(0, T):
        sum1 *= getPro(testSet7[i][j], Mean2[j], var2[j])
    sum2 = 1
    for j in range(0, T):
        sum2 *= getPro(testSet7[i][j], Mean3[j], var3[j])
    sum3 = 1
    for j in range(0, T):
        sum3 *= getPro(testSet7[i][j], Mean4[j], var4[j])
    sum4 = 1
    for j in range(0, T):
        sum4 *= getPro(testSet7[i][j], Mean5[j], var5[j])
    sum5 = 1
    for j in range(0, T):
        sum5 *= getPro(testSet7[i][j], Mean6[j], var6[j])
    sum6 = 1
    for j in range(0, T):
        sum6 *= getPro(testSet7[i][j], Mean7[j], var7[j])
    if (Pro7 * sum6 >= Pro1 * sum) & (Pro7 * sum6 >= Pro2 * sum1) & (Pro7 * sum6 >= Pro3 * sum2) & (
            Pro7 * sum6 >= Pro4 * sum3) & (Pro7 * sum6 >= Pro5 * sum4) & (Pro7 * sum6 >= Pro6 * sum5):
        add6 += 1
    else:
        add6 += 0
print("第六类正确数量(总数6)：")
print(add6)
print("accuracy:{:.2%}".format((add + add1 + add2 + add3 + add4 + add5 + add6) / (
        test1 + test2 + test3 + test4 + test5 + test6 + test7)))
