# from matrix_data import MatrixData
# from prettytable import PrettyTable
# from multiprocessing import Pool
# from strassen_algorithm import MatrixMultiplication
# import numpy as np
# import time

# class MatrixMultiplicationMultiProcess(object):

#     def __init__(self, pool) -> None:
#         """ Initialize the computation class, and the target dataset.
#         Args:
#             dataset: Unified group matrix dataset used for multiplication.
#         """
#         dataset = MatrixData().loaded_matrix_set
#         pool = pool
#         pool = Pool(pool)
#     def __getstate__(self):
#         self_dict = __dict__.copy()
#         del self_dict['pool']
#         return self_dict

#     def __setstate__(self, state):
#         __dict__.update(state)

#     def matrix_slicing(self, matrix):
#         """ Matrix slicing into 4 euqal parts.
#         """
#         matrix_size = len(matrix)
#         mid_matrix_size = matrix_size//2
#         x_11 = matrix[0:mid_matrix_size, 0:mid_matrix_size]
#         x_12 = matrix[0:mid_matrix_size, mid_matrix_size:matrix_size]
#         x_21 = matrix[mid_matrix_size:matrix_size, 0:mid_matrix_size]
#         x_22 = matrix[mid_matrix_size:matrix_size, mid_matrix_size:matrix_size]
#         return x_11, x_12, x_21, x_22
    
#     def multi_processing_matrix_multiply(self, matrix_1, matrix_2):
#         """ 
#         Multiply paired matrices with processing pool.
#         """
#         # Matrix slicing for each input matrixs, which need to split into 4 parts.
#         matrix_1 = matrix_1.astype(np.uint64)
#         matrix_2 = matrix_2.astype(np.uint64)

#         # Linear calculation without using numpy function. threhold
#         if matrix_1.shape[0] <= 4:
#             return np.dot(matrix_1, matrix_2)

#         # matrix_1 (x) slicing
#         x_11, x_12, x_21, x_22 = matrix_slicing(matrix=matrix_1)
#         # matrix_2 (y) slicing
#         y_11, y_12, y_21, y_22 = matrix_slicing(matrix=matrix_2)

#         m_1 = multi_processing_matrix_multiply, (x_11 + x_22, y_11 + y_22))
#         m_2 = multi_processing_matrix_multiply, (x_21 + x_22, y_11))
#         m_3 = multi_processing_matrix_multiply, (x_11, y_12 - y_22))
#         m_4 = multi_processing_matrix_multiply, (x_22, y_21 - y_11))
#         m_5 = multi_processing_matrix_multiply, (x_11 + x_12, y_22))
#         m_6 = multi_processing_matrix_multiply, (x_21 - x_11, y_11 + y_12))
#         m_7 = multi_processing_matrix_multiply, (x_12 - x_22, y_21 + y_22))

#         c_11 = m_1 + m_4 - m_5 + m_7
#         c_12 = m_3 + m_5
#         c_21 = m_2 + m_4
#         c_22 = m_1 + m_3 - m_2 + m_6

#         # Result concatation
#         leading_side = np.vstack((c_11, c_21))
#         trailing_side = np.vstack((c_12, c_22))
#         final_c = np.hstack((leading_side, trailing_side))

#         return final_c
    
#     def matrices_calculation_multi_process(self):
#         dataset = dataset
#         for groupIndex in range(10):
#             t = np.zeros(8, dtype=np.float64)
#             table = PrettyTable()
#             table.title = "Process time of G{} under sequential implementation".format(groupIndex)
#             table.field_names = ['Pair Index', 'Measured Sequential Time', 'Measured Cumulative Sequential Time']
#             for pairIndex in range(8):
#                 t_start = time.time()
#                 multi_processing_matrix_multiply(dataset[groupIndex][pairIndex][0], dataset[groupIndex][pairIndex][1])
#                 t_end = time.time()
#                 t_seconds = t_end-t_start
#                 t[pairIndex] = t_seconds
#             for pairIndex in range(8):
#                 table.add_row([pairIndex, t[pairIndex], sum(t[0:pairIndex+1])])
#             print(table)
    
#     def close_pool(self):
#         pool.close()
#         pool.join()

    

# # def main():
# #     ca2_strassen = MatrixMultiplicationMultiProcess()
# #     ca2_strassen.matrices_calculation_single_process()
# #     for pool in range(2, 9):
# #         ca2_strassen.matrices_calculation_multi_process()

# def test():
#     """
#     8 process match the Apple M1 chip 8 cores. For maxmize performance test.
#     """
#     ca2_strassen_test = MatrixMultiplicationMultiProcess(pool=8)
#     ca2_strassen_test.matrices_calculation_multi_process()
#     ca2_strassen_test.close_pool()



# if __name__ == "__main__":
#     test()
    
    
from matrix_data import MatrixData
from prettytable import PrettyTable
from multiprocessing import Pool
import multiprocessing
from strassen_algorithm import MatrixMultiplication
import numpy as np
import time


    # def initialize_pool(self):
    #     # Initialize the pool if it's not already created
    #     if pool is None:
    #         if multiprocessing.get_start_method() == 'spawn':
    #             # Use 'spawn' method to create the pool on Windows
    #             pool = Pool(pool)
    #         else:
    #             pool = Pool(pool, maxtasksperchild=1)

    
    
    

    
    # def close_pool(self):
    #     pool.close()
    #     pool.join()

# def test(pool):
#     """
#     8 process match the Apple M1 chip 8 cores. For maxmize performance test.
#     """
#     ca2_strassen_test = MatrixMultiplicationMultiProcess(pool=8)
#     ca2_strassen_test.matrices_calculation_multi_process(pool=pool)


def matrix_slicing(matrix):
    """ Matrix slicing into 4 euqal parts.
    """
    matrix_size = len(matrix)
    mid_matrix_size = matrix_size//2
    x_11 = matrix[0:mid_matrix_size, 0:mid_matrix_size]
    x_12 = matrix[0:mid_matrix_size, mid_matrix_size:matrix_size]
    x_21 = matrix[mid_matrix_size:matrix_size, 0:mid_matrix_size]
    x_22 = matrix[mid_matrix_size:matrix_size, mid_matrix_size:matrix_size]
    return x_11, x_12, x_21, x_22

def multi_processing_matrix_multiply(matrix_1, matrix_2):
    """ 
    Multiply paired matrices with processing pool.
    """
    # Matrix slicing for each input matrixs, which need to split into 4 parts.
    matrix_1 = matrix_1.astype(np.uint64)
    matrix_2 = matrix_2.astype(np.uint64)

    # Linear calculation without using numpy function. threhold
    if matrix_1.shape[0] == 1:
        return matrix_1*matrix_2
    
    # pool = Pool(pool)

    # matrix_1 (x) slicing
    x_11, x_12, x_21, x_22 = matrix_slicing(matrix=matrix_1)
    # matrix_2 (y) slicing
    y_11, y_12, y_21, y_22 = matrix_slicing(matrix=matrix_2)

    m_1 = multi_processing_matrix_multiply(x_11 + x_22, y_11 + y_22)
    m_2 = multi_processing_matrix_multiply(x_21 + x_22, y_11)
    m_3 = multi_processing_matrix_multiply(x_11, y_12 - y_22)
    m_4 = multi_processing_matrix_multiply(x_22, y_21 - y_11)
    m_5 = multi_processing_matrix_multiply(x_11 + x_12, y_22)
    m_6 = multi_processing_matrix_multiply(x_21 - x_11, y_11 + y_12)
    m_7 = multi_processing_matrix_multiply(x_12 - x_22, y_21 + y_22)

    c_11 = m_1 + m_4 - m_5 + m_7
    c_12 = m_3 + m_5
    c_21 = m_2 + m_4
    c_22 = m_1 + m_3 - m_2 + m_6

    # Result concatation
    leading_side = np.vstack((c_11, c_21))
    trailing_side = np.vstack((c_12, c_22))
    final_c = np.hstack((leading_side, trailing_side))

    # pool.close()
    # pool.join()

    return final_c

def entry_of_matrix_multi(matrix_1, matrix_2, number_of_process):
    # Matrix slicing for each input matrixs, which need to split into 4 parts.
    matrix_1 = matrix_1.astype(np.uint64)
    matrix_2 = matrix_2.astype(np.uint64)

    
    # matrix_1 (x) slicing
    x_11, x_12, x_21, x_22 = matrix_slicing(matrix=matrix_1)
    # matrix_2 (y) slicing
    y_11, y_12, y_21, y_22 = matrix_slicing(matrix=matrix_2)

    with Pool(number_of_process) as pool:
        m_1 = pool.apply_async(multi_processing_matrix_multiply, (x_11 + x_22, y_11 + y_22))
        m_2 = pool.apply_async(multi_processing_matrix_multiply, (x_21 + x_22, y_11))
        m_3 = pool.apply_async(multi_processing_matrix_multiply, (x_11, y_12 - y_22))
        m_4 = pool.apply_async(multi_processing_matrix_multiply, (x_22, y_21 - y_11))
        m_5 = pool.apply_async(multi_processing_matrix_multiply, (x_11 + x_12, y_22))
        m_6 = pool.apply_async(multi_processing_matrix_multiply, (x_21 - x_11, y_11 + y_12))
        m_7 = pool.apply_async(multi_processing_matrix_multiply, (x_12 - x_22, y_21 + y_22))

        pool.close()
        pool.join()

    c_11 = m_1.get() + m_4.get() - m_5.get() + m_7.get()
    c_12 = m_3.get() + m_5.get()
    c_21 = m_2.get() + m_4.get()
    c_22 = m_1.get() + m_3.get() - m_2.get() + m_6.get()

    # Result concatation
    leading_side = np.vstack((c_11, c_21))
    trailing_side = np.vstack((c_12, c_22))
    final_c = np.hstack((leading_side, trailing_side))

    

    return final_c

def matrices_calculation_multi_process(number_of_process, dataset):
    for groupIndex in range(10):
        t = np.zeros(8, dtype=np.float64)
        table = PrettyTable()
        table.title = "Process time of G{} under sequential implementation".format(groupIndex)
        table.field_names = ['Pair Index', 'Measured Sequential Time', 'Measured Cumulative Sequential Time']
        for pairIndex in range(8):
            t_start = time.time()
            entry_of_matrix_multi(dataset[groupIndex][pairIndex][0], dataset[groupIndex][pairIndex][1], number_of_process)
            t_end = time.time()
            t_seconds = t_end - t_start
            t[pairIndex] = t_seconds
        for pairIndex in range(8):
            table.add_row([pairIndex, t[pairIndex], sum(t[0:pairIndex + 1])])
        print(table)

if __name__ == "__main__":
    dataset = MatrixData().loaded_matrix_set
    matrices_calculation_multi_process(8, dataset)



        