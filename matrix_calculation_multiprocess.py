from matrix_data import MatrixData
from prettytable import PrettyTable
from multiprocessing import Pool
from strassen_algorithm import MatrixMultiplication
import numpy as np
import time

class MatrixMultiplicationMultiProcess(MatrixMultiplication):

    def __init__(self) -> None:
        """ Initialize the computation class, and the target dataset.
        Args:
            dataset: Unified group matrix dataset used for multiplication.
        """
        self.dataset = MatrixData().loaded_matrix_set

    def strassen_algorithm_multi_multiprocess(self, matrix_1, matrix_2):
        """ Fundamental strassen's algorithm for matrix multiplication with same 'nxn' size.
        """
        # Matrix slicing for each input matrixs, which need to split into 4 parts.
        matrix_1 = matrix_1.astype(np.uint64)
        matrix_2 = matrix_2.astype(np.uint64)

        # Linear calculation without using numpy function.
        if matrix_1.shape[0] == 1:
            return matrix_1*matrix_2

        # matrix_1 (x) slicing
        x_11, x_12, x_21, x_22 = self._strassen_algorithm_multi__matrix_slicing(matrix=matrix_1)
        # matrix_2 (y) slicing
        y_11, y_12, y_21, y_22 = self.__matrix_slicing(matrix=matrix_2)

        m_1 = self.strassen_algorithm_multi(x_11 + x_22, y_11 + y_22)
        m_2 = self.strassen_algorithm_multi(x_21 + x_22, y_11)
        m_3 = self.strassen_algorithm_multi(x_11, y_12 - y_22)
        m_4 = self.strassen_algorithm_multi(x_22, y_21 - y_11)
        m_5 = self.strassen_algorithm_multi(x_11 + x_12, y_22)
        m_6 = self.strassen_algorithm_multi(x_21 - x_11, y_11 + y_12)
        m_7 = self.strassen_algorithm_multi(x_12 - x_22, y_21 + y_22)

        c_11 = m_1 + m_4 - m_5 + m_7
        c_12 = m_3 + m_5
        c_21 = m_2 + m_4
        c_22 = m_1 + m_3 - m_2 + m_6

        # Result concatation
        leading_side = np.vstack((c_11, c_21))
        trailing_side = np.vstack((c_12, c_22))
        final_c = np.hstack((leading_side, trailing_side))

        return final_c

    def matrices_calculation_single_process(self):
        dataset = self.dataset

        for groupIndex in range(10):
            t = np.zeros(8, dtype=np.float64)
            table = PrettyTable()
            table.title = "Process time of G{} under sequential implementation".format(groupIndex)
            table.field_names = ['Pair Index', 'Measured Sequential Time', 'Measured Cumulative Sequential Time']
            for pairIndex in range(4):
                t_start = time.time()
                self.strassen_algorithm_multi_multiprocess(dataset[groupIndex][pairIndex][0], dataset[groupIndex][pairIndex][1])
                t_end = time.time()
                t_seconds = t_end-t_start
                t[pairIndex] = t_seconds
            for pairIndex in range(4):
                table.add_row([pairIndex, t[pairIndex], sum(t[0:pairIndex+1])])
            print(table)

def main():
    multi = MatrixMultiplicationMultiProcess()
    multi.matrices_calculation_single_process()


if __name__ == "__main__":
    main()
    
    



        