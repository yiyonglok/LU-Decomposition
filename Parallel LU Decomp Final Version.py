
# Parallel LU decomposition

import multiprocessing
import numpy
import random
import time


def row_operation_new(matrix_ik, L_ik, matrix_kk):
    matrix_row = matrix_ik - (L_ik * matrix_kk)
    return numpy.array(matrix_row)


if __name__ == "__main__":
    print("Parallel LU Decomposition\n")
    n = 100  # Square Matrix Dimensions
    random_int_high = 100

    print("Matrix Size:", n)

    matrix = numpy.random.randint(random_int_high, size=(n, n))
    print("A: \n", matrix)
    matrix = matrix.astype(float)

    L = numpy.zeros((n, n))
    # populate Lower Triangular Matrix
    for r in range(n):
        L[r, r] = 1.0

    pool = multiprocessing.Pool(multiprocessing.cpu_count() - 1)
    print("\nLogical Threads being used:", multiprocessing.cpu_count() - 1)

    start = time.time()

    # LU Decomp
    for k in range(n):
        if matrix[k, k] != 0:
            L[k+1:n, k] = matrix[k+1:n, k]/matrix[k, k] #Calculate Multipliers for the current column for rows below the diagonal 1 line

            matrix_ik_array = matrix[k + 1:, k:]
            L_ik_array = numpy.tile(numpy.array([L[k + 1:, k]]).transpose(), (1, n - k)) #Make the multiplier array the same shape for zipping
            matrix_kk_array = numpy.full((n - (k + 1), len(matrix[k, k:])), matrix[k, k:])

            values = list(zip(matrix_ik_array, L_ik_array, matrix_kk_array))

            return_value = pool.starmap(row_operation_new, values) #Starmap returns results in same order the tuples were sent to the function (verified via research)

            if return_value:
                matrix[k+1:, k:] = return_value #Replace the section of matrix that was updated
            values.clear()

    end = time.time()

    matrix = matrix.round(2)
    L = L.round(2)

    print("\nUpper:\n", matrix)
    print("Lower:\n", L)

    Time = end - start

    print("\nElapsed Time:", Time)