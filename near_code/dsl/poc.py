from .library_functions import AffineFunction

class POCAffine(AffineFunction):
    def __init__(self, input_size, output_size, num_units, name="Affine"):
        super().__init__(input_size, input_size, output_size, num_units, name)