##### PERFORMANCE COMPARISON


## Vectorization (GPU)

import numpy as np
import math

a = np.array([1,2,3,4])
print(a)

## Time Test:

import time
a = np.random.rand(10000000)
b = np.random.rand(10000000)


tic = time.time()
c = np.dot(a,b)
toc = time.time()

print("Vectorzied version: " + str(1000*(toc-tic)) + "ms")

## Vectorized version: 36.77082061767578ms

c = 0
tic = time.time()
for i in range(10000000):
        c += a[i]*b[i]
toc = time.time()        

print("For loop: " + str(1000*(toc-tic)) + "ms")


## For loop: 4471.627712249756ms


## Vectorized version is 125 times faster than For loop


#### VECTOR & MATRIX VALUED FUNCTIONS


### Non-vectorized approach

u = np.zeros((n,1))
for i in range(n):
        u[i] = math.exp(v[i])
        
               
### Vecorized approach

u = np.exp(v)        



#### BROADCASTING EXAMPLE

A = np.array([[56.0, 0.0, 4.4, 68.0],
              [1.2,104.0,52.0,8.0],
              [1.8,135.0,99.0,0.9]])

cal = A.sum(axis = 0)
print(cal)

## Percentages for each element. ".reshape(1,4)" called for visibility (not needed)
percentage = 100*A/cal.reshape(1,4)
print(percentage)


#### CODE BUGS - DON'T USE RANK 1 ARRAYS!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

a = np.random.randn(5)
print(a.shape) ## Rank 1 array - neither row vector, neither column vector
##[ 1.49025036  0.71112747 -1.53820941 -0.77516934 -1.5025448 ]

## Transpose won't work

print(a.T)
### [ 1.49025036  0.71112747 -1.53820941 -0.77516934 -1.5025448 ]

## Intstead of matrix:
print(np.dot(a,a.T))

## 7.95116499634

## For Neural Networks use: 
a = np.random.randn(5,1)  
print(a)      
#[[-0.39745663]
# [-1.40789752]
# [-1.92291323]
# [-0.61649198]
# [ 0.71956947]]

print(a.T)
# [[-0.39745663 -1.40789752 -1.92291323 -0.61649198  0.71956947]] ## two sqare brackets indicate that this is (1,5) matrix

print(np.dot(a,a.T))


#[[ 0.15797177  0.55957821  0.76427462  0.24502883 -0.28599766]
# [ 0.55957821  1.98217542  2.70726477  0.86795753 -1.01308008]
# [ 0.76427462  2.70726477  3.6975953   1.18546059 -1.38366967]
# [ 0.24502883  0.86795753  1.18546059  0.38006237 -0.44360881]
# [-0.28599766 -1.01308008 -1.38366967 -0.44360881  0.51778023]]


### Check shape via assert:
        
assert(a.shape == (5,1))        







