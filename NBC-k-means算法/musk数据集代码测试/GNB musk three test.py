import numpy as np
import math
from sklearn.naive_bayes import GaussianNB

# 连续型数据分类用正态分布公式
def getPro(theData, mean, var):
    pro = 1 / (math.sqrt(2 * math.pi) * math.sqrt(var)) * math.exp(-(theData - mean) ** 2 / (2 * var))
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

X = np.loadtxt('[018]musk01(0-1).txt')
# 其中有97
m = X.shape[1] - 1  # 属性数量
n = X.shape[0]  # 样本数目
T = 3
K = 2  # 类标记数量
# 主要过程：分组
# 去掉类标记
Class1 = 0
Class2 = 0

for i in X:
    if i[m] == 1:
        Class1 = Class1 + 1
    elif i[m] == 2:
        Class2 = Class2 + 1

train1 = int(Class1 * 0.5)
val1 = int(Class1 * 0.2)
test1 = Class1 - train1 - val1

train2 = int(Class2 * 0.5)
val2 = int(Class2 * 0.2)
test2 = Class2 - train2 - val2

# 随机产生多少个和为1的随机数W
G1 = [4, 12, 15, 17, 42, 55, 62, 69, 75, 93, 94, 128, 135, 144, 147, 162, 163] # 7
G2 = [0, 1, 2, 3, 5, 6, 8, 9, 10, 11, 13, 14, 16, 18, 19, 20, 21, 23, 24, 25, 26, 27, 29, 31, 33, 34, 35, 36, 37, 38, 39, 40, 41, 43, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 56, 57, 59, 60, 61, 63, 64, 65, 68, 70, 71, 73, 74, 76, 77, 78, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 107, 108, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 129, 130, 131, 132, 133, 136, 137, 138, 139, 140, 141, 142, 143, 148, 149, 151, 152, 153, 154, 155, 157, 158, 160, 161]# 3
G3 = [7, 22, 28, 30, 32, 44, 58, 66, 67, 72, 79, 106, 109, 110, 134, 145, 146, 150, 156, 159, 164, 165]
len1=len(G1)
len2=len(G2)
len3=len(G3)
l1=len1
l2=len1+len2
l3=len1+len2+len3

#随机训练集，验证集，测试集区

idx = np.random.choice(np.arange(Class1), size = train1, replace = False)
train_index1 = np.array(idx)
val_index1 = np.random.choice(np.delete(np.arange(Class1), train_index1), size = val1, replace = False)
test_index1 = np.delete(np.arange(Class1), np.append(train_index1, val_index1))

idx1 = np.random.choice(np.arange(Class2), size = train2, replace = False)
train_index2 = np.array(idx1)
val_index2 = np.random.choice(np.delete(np.arange(Class2), train_index2), size = val2, replace = False)
test_index2 = np.delete(np.arange(Class2), np.append(train_index2, val_index2))

print("train_index1 =",list(train_index1))
print("val_index1 =",list(val_index1))
print("test_index1 =",list(test_index1))
print("train_index2 =",list(train_index2))
print("val_index2 =",list(val_index2))
print("test_index2 =",list(test_index2))




#确认训练集，验证集，测试集区
train_index1 = [148, 145, 40, 10, 9, 84, 113, 155, 197, 91, 120, 138, 117, 56, 54, 22, 48, 151, 164, 150, 130, 99, 71, 23, 153, 8, 77, 64, 140, 69, 200, 11, 81, 106, 2, 166, 182, 70, 25, 80, 143, 33, 111, 5, 159, 110, 95, 46, 156, 32, 123, 188, 121, 161, 141, 51, 186, 124, 49, 3, 88, 27, 96, 7, 47, 35, 55, 185, 170, 169, 196, 128, 191, 206, 201, 195, 129, 171, 41, 168, 1, 204, 144, 26, 73, 177, 29, 89, 154, 158, 16, 38, 127, 104, 190, 181, 194, 193, 162, 19, 58, 125, 115]
val_index1 = [90, 203, 42, 21, 66, 198, 165, 98, 18, 45, 15, 87, 142, 50, 97, 78, 146, 132, 63, 137, 131, 173, 149, 157, 0, 205, 101, 172, 61, 4, 43, 176, 135, 163, 85, 83, 102, 34, 109, 119, 147]
test_index1 = [6, 12, 13, 14, 17, 20, 24, 28, 30, 31, 36, 37, 39, 44, 52, 53, 57, 59, 60, 62, 65, 67, 68, 72, 74, 75, 76, 79, 82, 86, 92, 93, 94, 100, 103, 105, 107, 108, 112, 114, 116, 118, 122, 126, 133, 134, 136, 139, 152, 160, 167, 174, 175, 178, 179, 180, 183, 184, 187, 189, 192, 199, 202]
train_index2 = [136, 223, 71, 149, 4, 264, 106, 63, 164, 171, 45, 54, 119, 35, 193, 16, 213, 165, 13, 96, 199, 101, 67, 261, 47, 156, 202, 82, 170, 166, 6, 140, 167, 112, 209, 188, 134, 79, 8, 250, 262, 128, 5, 137, 227, 155, 118, 74, 158, 93, 80, 179, 125, 7, 3, 206, 183, 66, 146, 173, 44, 92, 182, 42, 212, 122, 254, 102, 95, 23, 154, 37, 141, 32, 116, 34, 9, 160, 57, 127, 197, 58, 226, 241, 217, 150, 14, 91, 78, 30, 191, 229, 258, 28, 117, 195, 189, 153, 60, 178, 68, 65, 121, 89, 242, 73, 94, 186, 224, 69, 240, 22, 20, 214, 120, 247, 175, 100, 219, 232, 129, 131, 41, 50, 231, 87, 15, 194, 143, 208, 237, 174, 76, 144]
val_index2 = [27, 205, 103, 132, 139, 248, 177, 216, 211, 133, 185, 70, 192, 12, 138, 268, 0, 26, 252, 114, 148, 225, 236, 83, 56, 157, 207, 19, 123, 11, 52, 222, 81, 235, 255, 190, 176, 218, 49, 90, 97, 113, 38, 266, 55, 220, 204, 198, 10, 17, 99, 145, 36]
test_index2 = [1, 2, 18, 21, 24, 25, 29, 31, 33, 39, 40, 43, 46, 48, 51, 53, 59, 61, 62, 64, 72, 75, 77, 84, 85, 86, 88, 98, 104, 105, 107, 108, 109, 110, 111, 115, 124, 126, 130, 135, 142, 147, 151, 152, 159, 161, 162, 163, 168, 169, 172, 180, 181, 184, 187, 196, 200, 201, 203, 210, 215, 221, 228, 230, 233, 234, 238, 239, 243, 244, 245, 246, 249, 251, 253, 256, 257, 259, 260, 263, 265, 267]

W = getRandom(m * K) * 100
W=[0.0005275845932029088, 0.01718429917104686, 0.0027531601889124106, 0.012998597077541222, 0.06303502091299751, 0.0845944470207089, 0.0394126809680561, 0.001318724625626206, 0.0925366169320249, 0.009499347190709153, 0.02046005190914723, 0.013486279954787532, 0.01875975329205353, 0.0030130384598496284, 0.019877185853468572, 0.010924535472827108, 0.01712975560207825, 0.02957618944694056, 0.03015106311204739, 0.04119818007456971, 0.0016451498455075703, 0.03393003510967081, 0.01977261470627883, 0.020702706670792284, 0.017422398743737515, 0.011167013247221393, 0.04296469068333719, 0.012715430355477234, 0.025165524622402527, 0.026202410070652306, 0.02536154697652803, 0.06217309200797741, 0.048209840772460696, 0.03301045856645181, 0.10207013120149427, 0.028552078502866538, 0.015167118322154713, 0.02358677413666841, 0.06961430437044894, 0.005631349401928897, 0.07640500649182766, 0.007520586775865519, 0.0024630457269492964, 0.0063634214960004865, 0.030052278071175626, 0.0402465082442895, 0.001858271594136487, 0.11304928473270408, 0.09232440325472836, 0.022624307690309008, 0.041762816506062575, 0.024844903082008245, 0.027595693841557422, 0.0011481419445924777, 0.057097410870375745, 0.0500569120327532, 0.0009893212850122732, 0.0025108793936410565, 0.05481358691176084, 0.006018490254040495, 0.014600656385498451, 0.01734324450134541, 0.03696852244722435, 0.013167029846280973, 0.02742402336084412, 0.005222346360883069, 0.010185182372333672, 0.003393597569586635, 0.03209927979966267, 0.022291084130503062, 0.045440586580549974, 0.06222069214002459, 0.03239117895136051, 0.008158283584469019, 0.06218678340067806, 0.15368669056242995, 0.014558105254694942, 0.011069216723295583, 0.03372117203199292, 0.03775380946096049, 0.047686281874751706, 0.005966313477032349, 0.05235753520536363, 0.0491448190708947, 0.049610898658070085, 0.06922755716489276, 0.01757775731017276, 0.0008035265237428396, 0.03983949248878291, 0.03979958475142764, 0.0058118311166968535, 0.0038733614739140105, 0.005386586816028305, 0.00721086097613805, 0.0025594912688863382, 0.01645125563656094, 0.0033120619069399804, 0.09043176397796496, 0.06028775191615262, 0.03086959421647516, 0.01112058899595332, 0.004082007109861883, 0.014363639652260812, 0.009593741712561657, 0.011081728653881174, 0.061516906592957964, 0.0017403196054643, 0.06885185637618402, 0.006594700550598295, 0.07189591652396632, 0.0720520471169575, 0.08146561108223704, 0.0025118714106636296, 0.03406683436295388, 0.03219821974995354, 0.004983627052968214, 0.004829364986027866, 0.06694163472194273, 0.07923499057477189, 0.08952499444289747, 0.020033711586475638, 0.09155057009652207, 0.028138762180583105, 0.019773149194249846, 0.010379076191946812, 0.05408347362777673, 0.0003918960948606148, 0.00010132192567004405, 0.0466913098375145, 0.0005334460965317258, 0.027735805364110872, 0.005805938353842827, 0.14036693635914305, 0.004588877140805452, 0.10616443582234447, 0.005308609193695114, 0.050872301393299235, 0.009102975399831948, 0.01701758587189568, 0.03506903106331862, 0.033168172847581015, 0.04333260094470544, 0.005476686224117192, 0.007957932278392865, 0.02390292900391877, 0.02033632893105927, 0.012554969467923081, 0.07218139918698716, 0.03078179270897903, 0.0008114705293095817, 0.003921076473846013, 0.006721656199067204, 0.013009714700430615, 0.025147128931924743, 0.014822022881980073, 0.10313088389380326, 0.010471892353414482, 0.015268311862524152, 0.07525054198120495, 0.010330814999907758, 0.053579368260739696, 0.0354690904414162, 0.0042034635653188605, 0.057142487636395047, 0.003972427916710347, 0.06045983853343105, 0.025782565185205775, 0.0349490124014113, 0.0003472291630955969, 0.040637548288906034, 0.022795689957951254, 0.022152838984076974, 0.058307355764805446, 0.048375227202009055, 0.05360554921279531, 0.005271537293816724, 0.05783058974199882, 0.00039252559725301957, 0.09929245371992096, 0.01369785678042695, 0.06586474739907691, 0.003316342156098488, 0.056530812761118224, 0.009197367855449397, 0.040902019374045584, 0.018888099707091544, 0.06736224209703275, 0.002590176011543328, 0.010948508374334878, 0.006148230450296905, 0.05047914427955151, 0.009406267339825762, 0.01854950233315276, 0.001441242475026291, 0.007600051349459071, 0.048089398411433976, 0.028833334057893574, 0.039654233416650954, 0.019674497612453726, 0.035423691110009134, 0.009004815592518811, 0.03191964992098377, 0.01689233557155479, 0.018750893946362538, 0.00484006518362279, 0.02844144642935276, 0.03901795771469748, 0.005880721272462035, 0.003518688259455999, 0.1032872063460636, 0.10826153119373896, 0.013583223922482826, 0.006382629504102911, 0.009903490334617574, 0.01672300815112949, 0.025284395137352218, 0.04174852687098331, 0.019111320433325384, 0.0006437557218532239, 0.011023867676909743, 0.05743417156335885, 0.013108375531604296, 0.02127911531380223, 0.03247664238797217, 0.017976883886612005, 0.007577252342919635, 0.06859486982423627, 0.03148265077236122, 0.0012697564229009251, 0.039312378765424254, 0.04620467248297456, 0.07859736189028715, 0.014597974417807887, 0.04258735117126867, 0.11301987795282901, 0.02026371412110933, 0.007022208711421762, 0.05411432438368838, 0.0281123970546473, 0.11657406630529907, 0.01830374854933892, 0.05126493538269852, 0.035610054646081, 0.045757149263980286, 0.0066310165343972365, 0.010298655415993967, 0.014249741432423222, 0.023699991146584058, 0.006783633093608159, 0.10752584529093082, 0.03287641474258881, 0.03033149033775956, 0.03294153577313156, 0.007082442247649796, 0.07402668239927825, 0.015479675008295802, 0.13487910165418085, 0.01845394722877998, 0.004035037793757439, 0.015450552243655763, 0.024441624094343153, 0.008273368691908792, 0.020591125313969345, 0.001437354289510509, 0.0019436107409545392, 0.0017307416742724192, 0.009115002070818057, 0.0410125426213797, 0.07096167826382499, 0.012664546927581493, 0.023983442843402043, 0.012823965451800826, 0.050368488842841355, 0.012822372425744251, 0.0685506215210989, 0.014467091282776543, 0.0019423628298582424, 0.009558938155091644, 0.004838780503220701, 0.11440628794625404, 0.0028656877769624297, 0.005362684992537109, 0.025072550931148685, 0.011435773408277936, 0.04433148628542201, 0.0033930262388972114, 0.016502751324194452, 0.011396183657716328, 0.020368453664993465, 0.02166929764037577, 0.0411109217798166, 0.06604020951602671, 0.02096702454129068, 0.028473233299355978, 0.009761048652760309, 0.002456379981492316, 0.029026806327156207, 0.022354319822096225, 0.03258392164012383, 0.0037105121775553806, 0.03886081342017681, 9.22931214624e-05, 0.0014005594840292117, 0.02188912280315111, 0.010280849364550516, 0.018025219569521238, 0.039881962952543266, 0.02239194406665982, 0.051277699472146, 0.0032131733128637997, 0.0378222327342788, 0.015738809568862037, 0.044329882421825104, 0.022867905124936208, 0.08688850573028, 0.00340120338640343, 0.00869981161311235, 0.000705166878568103, 0.029779926366615975, 0.009592979268261454, 0.014984647071735807, 0.022991530985407016, 0.009720913166184402, 0.04124007146924373, 0.019175573585997602, 0.030707166587242287, 0.007052068048817934, 0.010856848419907868, 0.05904758220282891, 0.03746416608202275, 0.07762942215311552, 0.04022753838944903]


NewArray = np.ones((Class1, T + 1))
# 第0组
W1 = W[0:l1]
for i in range(0, Class1):
    add1 = 0
    for j in range(0, len1):
        add1 += W1[j] * X[i, G1[j]]
    NewArray[i][0] = add1
# 第1组
W2 = W[l1:l2]
for i in range(0, Class1):
    add2 = 0
    for j in range(0, len2):
        add2 += W2[j] * X[i, G2[j]]
    NewArray[i][1] = add2
# 第2组
W3 = W[l2:l3]
for i in range(0, Class1):
    add3 = 0
    for j in range(0, len3):
        add3 += W3[j] * X[i, G3[j]]
    NewArray[i][2] = add3
# 求类2的分组情况
NewArray1 = np.ones((Class2, T + 1)) * 2
# 第0组
W4 = W[l3:l3+l1]
for i in range(Class1, n):
    add1 = 0
    for j in range(0, len1):
        add1 += W4[j] * X[i, G1[j]]
    NewArray1[i - Class1][0] = add1
# 第1组
W5 = W[l1+l3:l3+l2]
for i in range(Class1, n):
    add2 = 0
    for j in range(0, len2):
        add2 += W5[j] * X[i, G2[j]]
    NewArray1[i - Class1][1] = add2
# 第2组
W6 = W[l3+l2:l3+l3]
for i in range(Class1, n):
    add3 = 0
    for j in range(0, len3):
        add3 += W6[j] * X[i, G3[j]]
    NewArray1[i - Class1][2] = add3

# print(NewArray1)
# 合并两个数组，得到真正的合并数据结果
NewArray = np.vstack((NewArray, NewArray1))
print(NewArray)

# 随机抽取样本训练集和测试集样本

X1 = NewArray[0:Class1, :]
X2 = NewArray[Class1:Class1 + Class2, :]

Data1 = X1[train_index1, :]
Data2 = X2[train_index2, :]
trainSet=np.vstack((Data1,Data2))
Y=trainSet[:,T]
trainSet=np.delete(trainSet,T,axis = 1)

testSet1 = np.delete(X1[test_index1, :], T, axis = 1)
testSet2 = np.delete(X2[test_index2, :], T, axis = 1)
trainSet1 = np.delete(Data1, T, axis = 1)
trainSet2 = np.delete(Data2, T, axis = 1)
valSet1=np.delete(X1[val_index1,:],T,axis = 1)
valSet2=np.delete(X2[val_index2,:],T,axis = 1)

# 求各类对应属性的均值和方差
Mean1 = np.mean(trainSet1, axis = 0)
Mean2 = np.mean(trainSet2, axis = 0)
# print(Mean2)
var1 = np.var(trainSet1, axis = 0)
var2 = np.var(trainSet2, axis = 0)



clf=GaussianNB()

clf.fit(trainSet,Y)

C1 = clf.predict(testSet1)
add = sum(C1 == 1)
print("第一类正确数量(总数):", test1)
print(add)
C2 = clf.predict(testSet2)
add1 = sum(C2 == 2)
print("第二类正确数量(总数)：", test2)
print(add1)

print("accuracy:{:.2%}".format((add + add1) / (test1+test2)))



