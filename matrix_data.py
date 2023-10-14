import numpy as np


class MatrixData:

    def __init__(self) -> None:
        """
        Args:
            loaded_matrix_set: The loaded_matrix_set is a three dimentional vector of 2D matrixs.
            The layout is: loaded_matrix_set[group index][pair index][paired matrix index]
            group index: [0:9]
            pair index: [0:7] -> [16:2048]
            paired matrix index: [0,1], first matrix, second matrix
        """
        self.loaded_matrix_set = self.__matrix_dataset_initialization('matrix_set.npy')

    def __matrix_dataset_initialization(self, path):
        """
        Private function for dataset initialization, return target dataset.
        """
        return np.load(path, allow_pickle=True)
        