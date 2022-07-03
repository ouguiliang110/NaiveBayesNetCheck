import random
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
def getPro(theData, mean, var):
    a=1
    if (mean == 0) & (var == 0):
        return a
    else:
        pro=1 / (math.sqrt(2 * math.pi) * math.sqrt(var)) * math.exp(-(theData - mean) ** 2 / (2 * var))
        return pro

X = np.loadtxt('[013]segment(0-1).txt')
X=X[:,np.append(np.arange(9,18),[X.shape[1]-1])]
m = X.shape[1]-1  # 属性数量
n = X.shape[0]  # 样本数目
K = 7  # 类标记数量
# 主要过程：分组
T = 3  # 分组数量

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

G1 = [4, 5, 8]  # 16
G2 = [2, 6, 7] # 2
G3 = [0, 1, 3]  # 1

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
test5 = Class5 - train5 - val5

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

# 确认数据集，验证集，测试集区
train_index1 = [183, 100, 180, 96, 31, 181, 49, 18, 307, 63, 68, 139, 156, 269, 268, 54, 216, 21, 256, 152, 8, 280, 89, 95, 124, 154, 66, 9, 270, 260, 33, 77, 164, 50, 245, 244, 43, 327, 38, 298, 144, 142, 23, 30, 91, 141, 162, 166, 58, 189, 80, 136, 325, 25, 15, 182, 163, 120, 215, 65, 119, 13, 207, 46, 6, 265, 293, 151, 71, 105, 281, 219, 155, 122, 186, 277, 169, 176, 324, 213, 279, 37, 4, 51, 16, 218, 311, 282, 259, 131, 83, 243, 284, 323, 203, 308, 191, 187, 150, 238, 109, 316, 193, 2, 1, 194, 28, 113, 110, 195, 92, 132, 248, 229, 328, 228, 292, 56, 40, 55, 69, 22, 318, 24, 118, 314, 266, 72, 116, 233, 143, 322, 242, 247, 241, 199, 137, 295, 85, 230, 309, 255, 221, 67, 208, 300, 158, 200, 102, 146, 278, 125, 264, 98, 320, 254, 52, 59, 212, 313, 87, 236, 53, 14, 149]
val_index1 = [258, 297, 7, 286, 171, 299, 111, 3, 90, 273, 153, 274, 246, 291, 205, 271, 106, 306, 267, 20, 160, 82, 64, 70, 192, 175, 27, 145, 12, 165, 29, 97, 75, 188, 19, 117, 232, 115, 0, 252, 301, 147, 127, 114, 99, 35, 231, 179, 310, 60, 129, 222, 108, 220, 47, 81, 45, 206, 251, 126, 224, 217, 178, 296, 211, 294]
test_index1 = [5, 10, 11, 17, 26, 32, 34, 36, 39, 41, 42, 44, 48, 57, 61, 62, 73, 74, 76, 78, 79, 84, 86, 88, 93, 94, 101, 103, 104, 107, 112, 121, 123, 128, 130, 133, 134, 135, 138, 140, 148, 157, 159, 161, 167, 168, 170, 172, 173, 174, 177, 184, 185, 190, 196, 197, 198, 201, 202, 204, 209, 210, 214, 223, 225, 226, 227, 234, 235, 237, 239, 240, 249, 250, 253, 257, 261, 262, 263, 272, 275, 276, 283, 285, 287, 288, 289, 290, 302, 303, 304, 305, 312, 315, 317, 319, 321, 326, 329]
train_index2 = [327, 229, 153, 305, 230, 295, 166, 48, 16, 116, 101, 255, 306, 253, 256, 150, 156, 175, 70, 184, 224, 217, 63, 80, 188, 44, 163, 307, 291, 29, 174, 55, 315, 75, 209, 10, 41, 314, 93, 329, 53, 15, 157, 303, 122, 112, 141, 139, 300, 220, 190, 170, 164, 144, 27, 279, 272, 128, 42, 274, 296, 192, 221, 115, 247, 201, 9, 82, 183, 211, 66, 252, 4, 78, 235, 106, 102, 130, 47, 202, 160, 17, 127, 154, 185, 246, 96, 57, 312, 323, 250, 104, 258, 203, 260, 7, 142, 5, 40, 282, 165, 294, 79, 311, 287, 318, 200, 83, 19, 148, 22, 131, 54, 172, 232, 193, 228, 322, 109, 146, 226, 171, 85, 123, 119, 207, 189, 213, 309, 136, 241, 181, 328, 324, 199, 273, 38, 195, 72, 26, 276, 61, 316, 161, 98, 69, 110, 21, 179, 28, 275, 124, 167, 12, 212, 155, 205, 219, 105, 223, 14, 90, 277, 298, 259]
val_index2 = [100, 222, 227, 231, 11, 321, 168, 1, 218, 8, 73, 113, 236, 58, 145, 204, 265, 137, 114, 159, 140, 86, 99, 269, 249, 118, 264, 65, 304, 67, 289, 51, 36, 32, 261, 151, 206, 60, 129, 317, 263, 280, 266, 176, 285, 326, 310, 23, 0, 81, 319, 20, 49, 215, 299, 35, 284, 292, 234, 108, 37, 43, 271, 313, 278, 237]
test_index2 = [2, 3, 6, 13, 18, 24, 25, 30, 31, 33, 34, 39, 45, 46, 50, 52, 56, 59, 62, 64, 68, 71, 74, 76, 77, 84, 87, 88, 89, 91, 92, 94, 95, 97, 103, 107, 111, 117, 120, 121, 125, 126, 132, 133, 134, 135, 138, 143, 147, 149, 152, 158, 162, 169, 173, 177, 178, 180, 182, 186, 187, 191, 194, 196, 197, 198, 208, 210, 214, 216, 225, 233, 238, 239, 240, 242, 243, 244, 245, 248, 251, 254, 257, 262, 267, 268, 270, 281, 283, 286, 288, 290, 293, 297, 301, 302, 308, 320, 325]
train_index3 = [17, 291, 68, 170, 132, 208, 167, 98, 327, 22, 290, 159, 80, 211, 34, 66, 288, 281, 300, 312, 39, 195, 106, 251, 293, 58, 24, 309, 18, 153, 262, 165, 228, 230, 117, 7, 20, 222, 26, 2, 213, 157, 169, 278, 269, 297, 266, 146, 240, 29, 129, 225, 70, 6, 56, 232, 217, 50, 69, 19, 321, 197, 121, 32, 241, 223, 158, 267, 35, 295, 137, 154, 99, 73, 207, 156, 257, 119, 88, 93, 203, 178, 52, 140, 189, 37, 77, 255, 96, 3, 231, 103, 224, 128, 196, 101, 188, 130, 51, 239, 305, 38, 164, 175, 109, 100, 151, 299, 268, 198, 136, 63, 323, 125, 234, 10, 286, 123, 102, 324, 168, 104, 107, 283, 237, 177, 194, 326, 160, 322, 285, 329, 284, 141, 86, 199, 192, 209, 148, 150, 114, 314, 143, 115, 247, 120, 112, 296, 280, 184, 294, 12, 147, 252, 204, 215, 110, 173, 31, 249, 313, 183, 308, 212, 45]
val_index3 = [161, 264, 185, 226, 244, 310, 258, 60, 205, 193, 90, 122, 55, 229, 201, 180, 113, 33, 1, 176, 221, 47, 42, 43, 54, 253, 25, 279, 307, 76, 218, 83, 138, 89, 274, 131, 94, 62, 202, 186, 318, 23, 317, 142, 85, 67, 65, 275, 44, 287, 174, 21, 191, 97, 91, 227, 216, 166, 210, 84, 28, 282, 182, 220, 13, 75]
test_index3 = [0, 4, 5, 8, 9, 11, 14, 15, 16, 27, 30, 36, 40, 41, 46, 48, 49, 53, 57, 59, 61, 64, 71, 72, 74, 78, 79, 81, 82, 87, 92, 95, 105, 108, 111, 116, 118, 124, 126, 127, 133, 134, 135, 139, 144, 145, 149, 152, 155, 162, 163, 171, 172, 179, 181, 187, 190, 200, 206, 214, 219, 233, 235, 236, 238, 242, 243, 245, 246, 248, 250, 254, 256, 259, 260, 261, 263, 265, 270, 271, 272, 273, 276, 277, 289, 292, 298, 301, 302, 303, 304, 306, 311, 315, 316, 319, 320, 325, 328]
train_index4 = [272, 313, 298, 270, 139, 63, 128, 215, 23, 241, 0, 45, 125, 189, 83, 40, 322, 111, 255, 261, 57, 199, 193, 108, 58, 203, 184, 186, 294, 259, 296, 54, 167, 221, 77, 8, 60, 309, 200, 210, 263, 156, 197, 224, 10, 284, 282, 195, 27, 163, 191, 196, 102, 56, 277, 265, 306, 85, 80, 243, 9, 251, 160, 79, 147, 303, 35, 235, 12, 17, 266, 292, 168, 325, 225, 262, 130, 323, 68, 281, 297, 165, 134, 24, 6, 146, 283, 123, 132, 112, 93, 183, 324, 293, 39, 316, 170, 229, 202, 61, 131, 192, 205, 246, 53, 179, 109, 129, 86, 78, 212, 91, 240, 51, 100, 187, 49, 75, 308, 321, 33, 99, 155, 30, 118, 228, 302, 121, 117, 71, 55, 185, 149, 84, 22, 260, 36, 21, 7, 310, 254, 104, 267, 164, 18, 38, 206, 291, 106, 204, 124, 234, 159, 98, 32, 138, 96, 142, 217, 271, 295, 329, 213, 137, 153]
val_index4 = [288, 94, 311, 73, 264, 274, 4, 198, 216, 276, 70, 175, 62, 141, 173, 69, 135, 88, 245, 257, 82, 25, 275, 300, 26, 3, 211, 105, 237, 287, 120, 150, 5, 299, 116, 66, 127, 31, 46, 90, 312, 76, 41, 285, 315, 220, 42, 47, 223, 194, 233, 227, 180, 214, 158, 161, 20, 97, 162, 182, 48, 34, 133, 154, 244, 13]
test_index4 = [1, 2, 11, 14, 15, 16, 19, 28, 29, 37, 43, 44, 50, 52, 59, 64, 65, 67, 72, 74, 81, 87, 89, 92, 95, 101, 103, 107, 110, 113, 114, 115, 119, 122, 126, 136, 140, 143, 144, 145, 148, 151, 152, 157, 166, 169, 171, 172, 174, 176, 177, 178, 181, 188, 190, 201, 207, 208, 209, 218, 219, 222, 226, 230, 231, 232, 236, 238, 239, 242, 247, 248, 249, 250, 252, 253, 256, 258, 268, 269, 273, 278, 279, 280, 286, 289, 290, 301, 304, 305, 307, 314, 317, 318, 319, 320, 326, 327, 328]
train_index5 = [164, 12, 8, 36, 253, 103, 191, 285, 147, 181, 175, 263, 278, 235, 105, 170, 10, 210, 25, 296, 286, 260, 14, 19, 217, 60, 297, 9, 34, 90, 176, 73, 106, 177, 206, 322, 55, 28, 316, 124, 78, 234, 23, 246, 283, 274, 319, 29, 239, 163, 258, 220, 277, 95, 213, 137, 197, 67, 269, 102, 326, 299, 46, 120, 279, 252, 308, 275, 22, 135, 257, 38, 323, 228, 204, 17, 65, 111, 139, 24, 162, 16, 167, 224, 219, 140, 294, 21, 250, 104, 190, 318, 196, 248, 223, 247, 86, 4, 321, 315, 267, 110, 100, 329, 72, 290, 298, 187, 180, 1, 328, 50, 122, 58, 201, 53, 74, 284, 45, 83, 304, 13, 59, 7, 128, 295, 171, 262, 117, 268, 80, 215, 240, 155, 92, 132, 141, 30, 311, 324, 207, 270, 129, 212, 101, 292, 66, 232, 0, 173, 145, 203, 241, 287, 119, 182, 32, 15, 156, 81, 11, 127, 6, 77, 71]
val_index5 = [154, 231, 245, 149, 218, 37, 198, 313, 289, 56, 317, 131, 152, 84, 200, 93, 98, 208, 320, 33, 205, 166, 160, 271, 265, 306, 52, 85, 70, 134, 244, 18, 146, 115, 79, 186, 256, 63, 211, 314, 174, 153, 192, 48, 43, 112, 168, 281, 199, 243, 225, 35, 327, 312, 221, 165, 310, 76, 47, 216, 27, 88, 2, 64, 148, 202]
test_index5 = [3, 5, 20, 26, 31, 39, 40, 41, 42, 44, 49, 51, 54, 57, 61, 62, 68, 69, 75, 82, 87, 89, 91, 94, 96, 97, 99, 107, 108, 109, 113, 114, 116, 118, 121, 123, 125, 126, 130, 133, 136, 138, 142, 143, 144, 150, 151, 157, 158, 159, 161, 169, 172, 178, 179, 183, 184, 185, 188, 189, 193, 194, 195, 209, 214, 222, 226, 227, 229, 230, 233, 236, 237, 238, 242, 249, 251, 254, 255, 259, 261, 264, 266, 272, 273, 276, 280, 282, 288, 291, 293, 300, 301, 302, 303, 305, 307, 309, 325]
train_index6 = [133, 322, 194, 109, 58, 166, 285, 137, 289, 93, 257, 254, 218, 14, 272, 53, 157, 261, 283, 135, 328, 59, 181, 190, 302, 275, 286, 165, 11, 12, 85, 51, 96, 153, 158, 56, 148, 281, 79, 30, 106, 199, 160, 5, 295, 288, 125, 43, 39, 75, 118, 246, 177, 204, 294, 321, 84, 226, 61, 23, 142, 300, 119, 31, 179, 48, 149, 42, 116, 244, 3, 87, 305, 44, 198, 147, 325, 202, 101, 269, 298, 162, 64, 280, 311, 174, 1, 35, 185, 126, 55, 70, 134, 161, 82, 132, 235, 168, 308, 193, 4, 72, 81, 9, 176, 131, 152, 41, 25, 78, 284, 26, 105, 208, 247, 18, 262, 111, 245, 220, 114, 222, 136, 91, 183, 95, 238, 191, 189, 303, 71, 206, 221, 282, 77, 138, 184, 140, 216, 46, 297, 323, 173, 130, 62, 304, 121, 108, 145, 129, 8, 277, 151, 268, 38, 186, 279, 17, 188, 209, 33, 267, 239, 230, 124]
val_index6 = [32, 155, 36, 233, 67, 232, 248, 57, 103, 69, 236, 187, 115, 40, 237, 203, 307, 313, 207, 310, 27, 159, 227, 92, 100, 102, 120, 88, 291, 34, 231, 170, 24, 318, 215, 195, 73, 127, 47, 90, 19, 309, 94, 68, 240, 37, 15, 223, 74, 86, 253, 107, 192, 172, 99, 252, 258, 83, 7, 242, 324, 271, 150, 60, 49, 112]
test_index6 = [0, 2, 6, 10, 13, 16, 20, 21, 22, 28, 29, 45, 50, 52, 54, 63, 65, 66, 76, 80, 89, 97, 98, 104, 110, 113, 117, 122, 123, 128, 139, 141, 143, 144, 146, 154, 156, 163, 164, 167, 169, 171, 175, 178, 180, 182, 196, 197, 200, 201, 205, 210, 211, 212, 213, 214, 217, 219, 224, 225, 228, 229, 234, 241, 243, 249, 250, 251, 255, 256, 259, 260, 263, 264, 265, 266, 270, 273, 274, 276, 278, 287, 290, 292, 293, 296, 299, 301, 306, 312, 314, 315, 316, 317, 319, 320, 326, 327, 329]
train_index7 = [60, 165, 302, 187, 32, 196, 175, 20, 44, 208, 87, 259, 168, 62, 268, 316, 126, 272, 2, 82, 292, 201, 0, 214, 279, 70, 52, 152, 185, 72, 238, 41, 218, 88, 109, 156, 293, 250, 35, 288, 31, 194, 77, 297, 314, 264, 162, 320, 176, 266, 103, 230, 210, 78, 284, 1, 25, 294, 155, 216, 258, 277, 317, 48, 42, 273, 69, 329, 97, 204, 142, 58, 40, 306, 188, 203, 13, 173, 202, 63, 270, 39, 121, 312, 225, 23, 132, 53, 15, 183, 26, 178, 235, 153, 170, 179, 98, 267, 275, 167, 159, 100, 220, 328, 269, 271, 209, 311, 143, 17, 141, 172, 33, 81, 231, 211, 28, 108, 9, 243, 80, 276, 102, 257, 192, 300, 186, 46, 21, 261, 74, 114, 115, 54, 305, 234, 219, 303, 85, 304, 67, 205, 191, 65, 75, 157, 198, 323, 240, 184, 37, 92, 124, 136, 104, 325, 181, 71, 118, 22, 146, 140, 166, 119, 283]
val_index7 = [236, 274, 249, 298, 244, 36, 247, 197, 171, 91, 318, 6, 262, 113, 291, 222, 7, 134, 149, 189, 24, 255, 43, 161, 148, 308, 139, 319, 199, 309, 94, 125, 289, 59, 154, 310, 195, 112, 45, 105, 326, 128, 290, 169, 147, 313, 95, 215, 321, 324, 66, 38, 101, 229, 29, 287, 212, 57, 12, 122, 206, 286, 18, 224, 280, 160]
test_index7 = [3, 4, 5, 8, 10, 11, 14, 16, 19, 27, 30, 34, 47, 49, 50, 51, 55, 56, 61, 64, 68, 73, 76, 79, 83, 84, 86, 89, 90, 93, 96, 99, 106, 107, 110, 111, 116, 117, 120, 123, 127, 129, 130, 131, 133, 135, 137, 138, 144, 145, 150, 151, 158, 163, 164, 174, 177, 180, 182, 190, 193, 200, 207, 213, 217, 221, 223, 226, 227, 228, 232, 233, 237, 239, 241, 242, 245, 246, 248, 251, 252, 253, 254, 256, 260, 263, 265, 278, 281, 282, 285, 295, 296, 299, 301, 307, 315, 322, 327]

class PSO:
    def __init__(self, parameters):

        self.NGEN = parameters[0]  # 迭代的代数
        self.pop_size = parameters[1]  # 种群大小
        self.var_num = m * K  # 变量个数
        self.pop_x = np.random.dirichlet(np.ones(self.var_num),size = pop_size)*10
        self.pop_v = np.zeros((self.pop_size, self.var_num))
        self.p_best = np.zeros((self.pop_size, self.var_num))  # 每个粒子最优的位置
        self.g_best = np.zeros((1, self.var_num))  # 全局最优的位置

        # 初始化第0代初始全局最优解
        temp = -1
        for i in range(self.pop_size):
            for j in range(self.var_num):
                self.pop_v[i][j] = random.uniform(0, 1)
            self.p_best[i] = self.pop_x[i]  # 存储最优的个体
            fit = self.fitness(self.p_best[i])
            if fit > temp:
                self.g_best = self.p_best[i]
                temp = fit

    def fitness(self, W):
        # 求类1的分组情况
        # 求类1的分组情况
        NewArray1 = np.ones((Class1, T + 1))
        # 第0组
        W1 = W[0:3]
        for i in range(0, Class1):
            add1 = 0
            for j in range(0, 3):
                add1 += W1[j] * X[i, G1[j]]
            NewArray1[i][0] = add1
        # 第1组
        W2 = W[3:6]
        for i in range(0, Class1):
            add2 = 0
            for j in range(0, 3):
                add2 += W2[j] * X[i, G2[j]]
            NewArray1[i][1] = add2
        # 第2组
        W3 = W[6:9]
        for i in range(0, Class1):
            add3 = 0
            for j in range(0, 3):
                add3 += W3[j] * X[i, G3[j]]
            NewArray1[i][2] = add3
        # print(NewArray1)

        # 求类2的分组情况
        NewArray2 = np.ones((Class2, T + 1)) * 2
        # 第0组
        W4 = W[9:12]
        for i in range(Class1, Class1 + Class2):
            add1 = 0
            for j in range(0, 3):
                add1 += W4[j] * X[i, G1[j]]
            NewArray2[i - Class1][0] = add1
        # 第1组
        W5 = W[12:15]
        for i in range(Class1, Class1 + Class2):
            add2 = 0
            for j in range(0, 3):
                add2 += W5[j] * X[i, G2[j]]
            NewArray2[i - Class1][1] = add2
        # 第2组
        W6 = W[15:18]
        for i in range(Class1, Class1 + Class2):
            add3 = 0
            for j in range(0, 3):
                add3 += W6[j] * X[i, G3[j]]
            NewArray2[i - Class1][2] = add3
        # print(NewArray2)

        # 求类3的分组情况
        NewArray3 = np.ones((Class3, T + 1)) * 3
        # 第0组
        W7 = W[18:21]
        for i in range(Class1 + Class2, Class1 + Class2 + Class3):
            add1 = 0
            for j in range(0, 3):
                add1 += W7[j] * X[i, G1[j]]
            NewArray3[i - Class1 - Class2][0] = add1
        # 第1组
        W8 = W[21:24]
        for i in range(Class1 + Class2, Class1 + Class2 + Class3):
            add2 = 0
            for j in range(0, 3):
                add2 += W8[j] * X[i, G2[j]]
            NewArray3[i - Class1 - Class2][1] = add2
        # 第2组
        W9 = W[24:27]
        for i in range(Class1 + Class2, Class1 + Class2 + Class3):
            add3 = 0
            for j in range(0, 3):
                add3 += W9[j] * X[i, G3[j]]
            NewArray3[i - Class1 - Class2][2] = add3
        # print(NewArray3)

        # 求类4的分组情况
        NewArray4 = np.ones((Class4, T + 1)) * 4
        # 第0组
        W10 = W[27:30]
        for i in range(Class1 + Class2 + Class3, Class1 + Class2 + Class3 + Class4):
            add1 = 0
            for j in range(0, 3):
                add1 += W10[j] * X[i, G1[j]]
            NewArray4[i - Class1 - Class2 - Class3][0] = add1
        # 第1组
        W11 = W[30:33]
        for i in range(Class1 + Class2 + Class3, Class1 + Class2 + Class3 + Class4):
            add2 = 0
            for j in range(0, 3):
                add2 += W11[j] * X[i, G2[j]]
            NewArray4[i - Class1 - Class2 - Class3][1] = add2
        # 第2组
        W12 = W[33:36]
        for i in range(Class1 + Class2 + Class3, Class1 + Class2 + Class3 + Class4):
            add3 = 0
            for j in range(0, 3):
                add3 += W12[j] * X[i, G3[j]]
            NewArray4[i - Class1 - Class2 - Class3][2] = add3
        # print(NewArray4)

        # 求类5的分组情况
        NewArray5 = np.ones((Class5, T + 1)) * 5
        # 第0组
        W13 = W[36:39]
        for i in range(Class1 + Class2 + Class3 + Class4, Class1 + Class2 + Class3 + Class4 + Class5):
            add1 = 0
            for j in range(0, 3):
                add1 += W13[j] * X[i, G1[j]]
            NewArray5[i - Class1 - Class2 - Class3 - Class4][0] = add1
        # 第1组
        W14 = W[39:42]
        for i in range(Class1 + Class2 + Class3 + Class4, Class1 + Class2 + Class3 + Class4 + Class5):
            add2 = 0
            for j in range(0, 3):
                add2 += W14[j] * X[i, G2[j]]
            NewArray5[i - Class1 - Class2 - Class3 - Class4][1] = add2
        # 第2组
        W15 = W[42:45]
        for i in range(Class1 + Class2 + Class3 + Class4, Class1 + Class2 + Class3 + Class4 + Class5):
            add3 = 0
            for j in range(0, 3):
                add3 += W15[j] * X[i, G3[j]]
            NewArray5[i - Class1 - Class2 - Class3 - Class4][2] = add3
        # print(NewArray5)

        # 求类6的分组情况
        NewArray6 = np.ones((Class6, T + 1)) * 6
        # 第0组
        W16 = W[45:48]
        for i in range(Class1 + Class2 + Class3 + Class4 + Class5, Class1 + Class2 + Class3 + Class4 + Class5 + Class6):
            add1 = 0
            for j in range(0, 3):
                add1 += W16[j] * X[i, G1[j]]
            NewArray6[i - Class1 - Class2 - Class3 - Class4 - Class5][0] = add1
        # 第1组
        W17 = W[48:51]
        for i in range(Class1 + Class2 + Class3 + Class4 + Class5, Class1 + Class2 + Class3 + Class4 + Class5 + Class6):
            add2 = 0
            for j in range(0, 3):
                add2 += W17[j] * X[i, G2[j]]
            NewArray6[i - Class1 - Class2 - Class3 - Class4 - Class5][1] = add2
        # 第2组
        W18 = W[51:54]
        for i in range(Class1 + Class2 + Class3 + Class4 + Class5, Class1 + Class2 + Class3 + Class4 + Class5 + Class6):
            add3 = 0
            for j in range(0, 3):
                add3 += W18[j] * X[i, G3[j]]
            NewArray6[i - Class1 - Class2 - Class3 - Class4 - Class5][2] = add3

        # 求类7的分组情况
        NewArray7 = np.ones((Class6, T + 1)) * 7
        # 第0组
        W19 = W[54:57]
        for i in range(num6, num7):
            add1 = 0
            for j in range(0, 3):
                add1 += W19[j] * X[i, G1[j]]
            NewArray7[i - num6][0] = add1
        # 第1组
        W20 = W[57:60]
        for i in range(num6, num7):
            add2 = 0
            for j in range(0, 3):
                add2 += W20[j] * X[i, G2[j]]
            NewArray7[i - num6][1] = add2
        # 第2组
        W21 = W[60:63]
        for i in range(num6, num7):
            add3 = 0
            for j in range(0, 3):
                add3 += W21[j] * X[i, G3[j]]
            NewArray7[i - num6][2] = add3
        # print(NewArray1)
        # 合并两个数组，得到真正的合并数据结果
        NewArray = np.vstack((NewArray1, NewArray2, NewArray3, NewArray4, NewArray5, NewArray6, NewArray7))
        # print(NewArray)
        # 随机抽取样本训练集和测试集样本
        # print(X)
        X1 = NewArray[0:Class1, :]
        X2 = NewArray[Class1:num2, :]
        X3 = NewArray[num2:num3, :]
        X4 = NewArray[num3:num4, :]
        # print(X4)
        X5 = NewArray[num4:num5, :]
        # print(X5)
        X6 = NewArray[num5:num6, :]
        X7 = NewArray[num6:num7, :]

        Data1 = X1[train_index1, :]
        Data2 = X2[train_index2, :]
        Data3 = X3[train_index3, :]
        Data4 = X4[train_index4, :]
        Data5 = X5[train_index5, :]
        Data6 = X6[train_index6, :]
        Data7 = X7[train_index7, :]


        trainSet = np.vstack((Data1, Data2, Data3, Data4, Data5, Data6, Data7))
        Y = trainSet[:, T]
        trainSet = np.delete(trainSet, T, axis = 1)


        valSet1 = np.delete(X1[test_index1, :], T, axis = 1)
        valSet2 = np.delete(X2[test_index2, :], T, axis = 1)
        valSet3 = np.delete(X3[test_index3, :], T, axis = 1)
        valSet4 = np.delete(X4[test_index4, :], T, axis = 1)
        valSet5 = np.delete(X5[test_index5, :], T, axis = 1)
        valSet6 = np.delete(X6[test_index6, :], T, axis = 1)
        valSet7 = np.delete(X7[test_index7, :], T, axis = 1)

        trainSet1 = np.delete(Data1, T, axis = 1)
        trainSet2 = np.delete(Data2, T, axis = 1)
        trainSet3 = np.delete(Data3, T, axis = 1)
        trainSet4 = np.delete(Data4, T, axis = 1)
        trainSet5 = np.delete(Data5, T, axis = 1)
        trainSet6 = np.delete(Data6, T, axis = 1)
        trainSet7 = np.delete(Data7, T, axis = 1)

        valSet1 = np.delete(X1[val_index1, :], T, axis = 1)
        valSet2 = np.delete(X2[val_index2, :], T, axis = 1)
        valSet3 = np.delete(X3[val_index3, :], T, axis = 1)
        valSet4 = np.delete(X4[val_index4, :], T, axis = 1)
        valSet5 = np.delete(X5[val_index5, :], T, axis = 1)
        valSet6 = np.delete(X6[val_index6, :], T, axis = 1)
        valSet7 = np.delete(X7[val_index7, :], T, axis = 1)


        allval=val1+val2+val3+val4+val5+val6+val7
        alltest=test1+test2+test3+test4+test5+test6+test7
        alltrain = train1 + train2 + train3 + train4 + train5 + train6 + train7

        clf = GaussianNB()

        clf.fit(trainSet, Y)

        C1 = clf.predict(valSet1)
        add = sum(C1 == 1)
       # print(add)
        C2 = clf.predict(valSet2)
        add1 = sum(C2 == 2)
       # print(add1)
        C3 = clf.predict(valSet3)
        add2 = sum(C3 == 3)
        #print(add2)
        C4 = clf.predict(valSet4)
        add3 = sum(C4 == 4)
        #print(add3)
        C5 = clf.predict(valSet5)
        add4 = sum(C5 == 5)
        #print(add4)
        C6 = clf.predict(valSet6)
        add5 = sum(C6 == 6)
        #print(add5)
        C7 = clf.predict(valSet7)
        add6 = sum(C7 == 7)
        #print(add6)

        acc = (add + add1 + add2 + add3 + add4 + add5 + add6) / (allval)
        return acc

    def update_operator(self, pop_size):
        c1 = 2
        c2 = 2
        w = 0.4
        for i in range(pop_size):
            # 更新速度
            self.pop_v[i] = w * self.pop_v[i] + c1 * random.uniform(0, 1) * abs(
                    self.p_best[i] - self.pop_x[i]) + c2 * random.uniform(0, 1) * abs(self.g_best - self.pop_x[i])
            # 更新位置
            self.pop_x[i] = self.pop_x[i] + self.pop_v[i]
            # 越界保护

            # 更新p_best和g_best
            if self.fitness(self.pop_x[i]) > self.fitness(self.p_best[i]):
                self.p_best[i] = self.pop_x[i]
            if self.fitness(self.pop_x[i]) > self.fitness(self.g_best):
                self.g_best = self.pop_x[i]

    def main(self):
        popobj = []
        self.ng_best=np.random.dirichlet(np.ones(m * K), size = 1)*10
        self.ng_best=self.ng_best.flatten()
        for gen in range(self.NGEN):
            self.update_operator(self.pop_size)
            popobj.append(self.fitness(self.g_best))
            print('############ Generation {} ############'.format(str(gen + 1)))
            if self.fitness(self.g_best) > self.fitness(self.ng_best):
                self.ng_best = self.g_best.copy()
            print('最好的位置：{}'.format(self.ng_best))
            print(list(self.ng_best))
            print('最大的函数值：{}'.format(self.fitness(self.ng_best)))
        print("---- End of (successful) Searching ----")

        plt.figure()
        plt.title("Figure1")
        plt.xlabel("iterators", size = 14)
        plt.ylabel("val-accuracy", size = 14)
        t = [t for t in range(self.NGEN)]
        plt.plot(t, popobj, color = 'b', linewidth = 2)
        plt.show()


if __name__ == '__main__':
    NGEN = 5
    pop_size = 200
    parameters = [NGEN, pop_size]
    pso = PSO(parameters)
    pso.main()
