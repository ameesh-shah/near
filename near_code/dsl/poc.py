from .library_functions import AffineFunction, LibraryFunction
import torch.nn as nn
import torch

if torch.cuda.is_available():
    device = 'cuda:0'
else:
    device = 'cpu'

class POCAffine(AffineFunction):
    def __init__(self, input_size, output_size, num_units, name="POCAffine"):
        super().__init__(input_size, input_size, output_size, num_units, name)