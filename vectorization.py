## Vectorization (GPU)

import numpy as np

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

