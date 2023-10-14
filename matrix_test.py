from matrix_data import MatrixData
import numpy as np
from prettytable import PrettyTable

class MatrixMultiplication:

    def __init__(self) -> None:
        self.dataset = MatrixData().loaded_matrix_set

    def matrix_test(self):
        matrix = self.dataset
        print(matrix[0][0][0])
        print(len(matrix[0][0][1]))

if __name__ == "__main__":
    table = PrettyTable()
    table.title = "Process time of G{} under sequential implementation".format(0)
    table.field_names = ['Pair Index', 'Measured Sequential Time', 'Measured Cumulative Sequential Time']
    table.add_row([0, 0, 0])
    print(table)