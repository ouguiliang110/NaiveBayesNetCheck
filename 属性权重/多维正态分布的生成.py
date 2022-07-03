import numpy as np
import scipy.stats as at
from sko.PSO import PSO
np.set_printoptions(precision = 4)
mean=(1,2)
cov =[[1,0.8],[0.8,4]]
num=50
x=np.random.multivariate_normal(mean, cov,(1,num),'raise')
print(x)
print(x[0])
x=x[0]
f=at.multivariate_normal.pdf(x,mean=mean,cov = cov)
print(f)

f1=at.multivariate_normal.pdf(x[:,0], mean = 1, cov = 1)

f2=at.multivariate_normal.pdf(x[:,1], mean=2,cov=2)

f_mult=f1*f2
print(f_mult)

def fitness(w):
    w1,w2=w
    add=0
    for i in range(num):
        add+=(f[i]-(f1[i]**w1)*(f2[i]**w2))**2
    return add
pso=PSO(func= fitness,dim=2,lb = [-5]*2,ub=[5]*2)
pso.run()
W=pso.gbest_x
print(pso.gbest_x)
print(pso.gbest_y)
f3=[]
for i in range(num):
    f3.append(f1[i]**W[0]*f2[i]**W[1])
print("边缘概率密度F(x1)",f1)
print("边缘概率密度F(x2)",f2)
print("联合概率密度",np.array(f))
print("边缘概率密度乘积",f_mult)
print("加权边缘概率密度",np.array(f3))
