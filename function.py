import numpy
import matplotlib.pyplot as plt

val=numpy.zeros(10).reshape(2,5) #type: numpy.ndarray
val1=numpy.eye(10)               #type: numpy.ndarray
val3=numpy.ones((3,4))
val6=numpy.arange(100).reshape(2,5,10)
val67=val6[:,1:3,:]              #type: numpy.ndarray
#b=numpy.array([False,True,False,True])
#val5=val67[b,:]
print(val67.shape)
data=numpy.array([1,-2,-2,4,3,4,-3])
data.sort()
data[data<0]=0
print(data)
num=numpy.arange(16).reshape(4,4)    #type: numpy.ndarray
v=numpy.linalg.eig(num)
arr=numpy.random.randn(4,4)
arr1=numpy.where(arr>0,2,-2)
mean=arr1.mean(axis=0)
def logistic(z):
    return 1/(1+numpy.exp(-z))
z=numpy.linspace(-6,6,100)
plt.plot(z,logistic(z),'b')
plt.xlabel('$z$', fontsize=15)
plt.ylabel('$\sigma(z)$', fontsize=15)
plt.title('logistic function')
plt.grid()
plt.show()
hehe=numpy.arange(3)
hehe1=hehe[:-1]
print(hehe)
print(hehe1)




#hehe=[numpy.random.randn(y,1) for y in range(10)])


