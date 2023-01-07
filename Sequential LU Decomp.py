
# Sequential LU decomposition

import numpy
import random
import time


print("Sequential LU Decomposition\n")

n = 100 #Square Matrix Dimensions
random_int_high = 100

print("Matrix Size:", n)

matrix = numpy.random.randint(random_int_high, size=(n, n))
print("A: \n", matrix)
matrix = matrix.astype(float)

L = numpy.zeros((n, n))
# populate Lower Triangular Matrix
for r in range(n):
    L[r,r] = 1.0

start = time.time()

#LU Decomp
for k in range(n):
    if matrix[k, k] != 0:
        for i in range(k + 1, n):
            L[i, k] = matrix[i, k] / matrix[k, k] # calculate the multipliers for current column
            for j in range(k, n):
                #print("spot:",i,",",j)
                matrix[i, j] = matrix[i, j] - (L[i, k] * matrix[k, j])  # apply transformation to remaining submatrix

end = time.time()

matrix = matrix.round(2)
L = L.round(2)

print("\nUpper:\n", matrix)
print("Lower:\n", L)

Time = end - start

print("\nElapsed Time:", Time)