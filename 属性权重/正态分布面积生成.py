import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
norm.cdf(1.96)

def solve(m1,m2,std1,std2):
  a = 1/(2*std1**2) - 1/(2*std2**2)
  b = m2/(std2**2) - m1/(std1**2)
  c = m1**2 /(2*std1**2) - m2**2 / (2*std2**2) - np.log(std2/std1)
  return np.roots([a,b,c])

plt.rcParams['font.sans-serif']=['Microsoft YaHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False #用来正常显示负号

m1=0.273
std1=0.043
m2=0.342
std2=0.050
m3=0.401
std3=0.058


#Get point of intersect
result = solve(m1,m2,std1,std2)

#Get point on surface
x = np.linspace(-0.25,0.8,10000)
plot1=plt.plot(x,norm.pdf(x,m1,std1),label = "类1",color='tab:blue')
plot2=plt.plot(x,norm.pdf(x,m2,std2),label = "类2",color='tab:orange')
plot3=plt.plot(result,norm.pdf(result,m1,std1),'o')

plt.legend()
#Plots integrated area
r = result[0]
olap = plt.fill_between(x[x>r], 0, norm.pdf(x[x>r],m1,std1),alpha=0.3,color='tab:blue',label="类1")
olap = plt.fill_between(x[x<r], 0, norm.pdf(x[x<r],m2,std2),alpha=0.3,color='tab:orange',label="类2")

# integrate
area = norm.cdf(r,m2,std2) + (1.-norm.cdf(r,m1,std1))
plt.title("RM-WNBC(相交区域面积="+str(format(area, '.3f'))+")")
print("Area under curves", area)

plt.show()


#Get point of intersect
result = solve(m1,m3,std1,std3)

#Get point on surface
x = np.linspace(-0.15,0.8,10000)
plot1=plt.plot(x,norm.pdf(x,m1,std1),label = "类1",color='tab:blue')
plot2=plt.plot(x,norm.pdf(x,m3,std3),label = "类3",color='tab:green')
plot3=plt.plot(result,norm.pdf(result,m1,std1),'o')

plt.legend()
#Plots integrated area
r = result[0]
olap = plt.fill_between(x[x>r], 0, norm.pdf(x[x>r],m1,std1),alpha=0.3,color='tab:blue',label="类1")
olap = plt.fill_between(x[x<r], 0, norm.pdf(x[x<r],m3,std3),alpha=0.3,color='tab:green',label="类2")

# integrate
area = norm.cdf(r,m3,std3) + (1.-norm.cdf(r,m1,std1))
plt.title("RM-WNBC(相交区域面积="+str(format(area, '.3f'))+")")
print("Area under curves", area)

plt.show()


#Get point of intersect
result = solve(m2,m3,std2,std3)

#Get point on surface
x = np.linspace(-0.1,0.8,10000)
plot1=plt.plot(x,norm.pdf(x,m2,std2),label = "类2",color='tab:orange')
plot2=plt.plot(x,norm.pdf(x,m3,std3),label = "类3",color='tab:green')
plot3=plt.plot(result,norm.pdf(result,m2,std2),'o')

plt.legend()
#Plots integrated area
r = result[0]
olap = plt.fill_between(x[x>r], 0, norm.pdf(x[x>r],m2,std2),alpha=0.3,color='tab:orange',label="类1")
olap = plt.fill_between(x[x<r], 0, norm.pdf(x[x<r],m3,std3),alpha=0.3,color='tab:green',label="类2")

# integrate
area = norm.cdf(r,m3,std3) + (1.-norm.cdf(r,m2,std2))
plt.title("RM-WNBC(相交区域面积="+str(format(area, '.3f'))+")")
print("Area under curves", area)

plt.show()