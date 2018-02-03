import numpy
import pandas as pd

val=[1,2,3,4,5,6]
data=numpy.array(val)
print(data)
val1=numpy.zeros(10)
print(val1)
val2=numpy.zeros((3,4))
print(val2)
a=numpy.random.rand(10,10)
print(a)
b=numpy.random.rand(10,10)
c=numpy.dot(a,b)
print(c)
def aha(n):
    return numpy.zeros(n)

def func(i,j):
    return (i+1)*(j+1)

d=aha(300)
print(d)

n=numpy.fromfunction(func,(3,3))
n1=n[:,1:3]
print(n1)
arr=numpy.arange(0,15).reshape(3,5)
assert isinstance(arr,numpy.ndarray)
arr1=arr.T
arr1.reshape(3,5)
arr2=numpy.zeros((3,5),dtype=int)
assert isinstance(arr2,numpy.ndarray)
arr3=numpy.vstack((arr,arr2))
arr4=arr3[1:4,:]
arr5=arr4
arr5.shape=(5,3)
print(arr4.shape)
ase=numpy.arange(12)
p=ase.reshape(3,4)
assert isinstance(p,numpy.ndarray)
de=numpy.ones((3,4))
arr2=arr3**2
arr6=numpy.eye(3)

test=pd.Series([1,2,1,2,3,4,2,3])
dad=pd.cut(test,4)
