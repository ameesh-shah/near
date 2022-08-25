from .library_functions import LibraryFunction, ITE, SimpleITE, init_neural_function
from .mujocoant import AntAffineFunction
import torch.nn as nn
import torch.nn.functional as F
import torch

if torch.cuda.is_available():
    device = 'cuda:0'
else:
    device = 'cpu'

class POCAffine(AntAffineFunction):
    def __init__(self, input_size, output_size, num_units, name="POCAffineFunction"):
        super().__init__(input_size, input_size, output_size, num_units)
        self.name = name
    
    
class POCITE1(SimpleITE):
    def __init__(self, input_type, output_type, input_size, output_size, num_units, eval_function=None, function1=None, function2=None, beta=1.0, name="POCITE1"):
        super().__init__(input_type, output_type, input_size, output_size, num_units, eval_function, function1, function2, beta)
        self.name = name
        
        
class Binarizer2(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return (input >= 0).to(dtype=torch.float32)#, 1

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input *= F.relu(1-input.abs())
        return grad_input

class POCITE2(LibraryFunction):
    """(Smoothed) If-The-Else."""

    def __init__(self, input_type, output_type, input_size, output_size, num_units, eval_function=None, function1=None, function2=None, beta=1.0, name="POCITE2"):
        if eval_function is None:
            eval_function = init_neural_function(input_type, "atom", input_size, 1, num_units)
        if function1 is None:
            function1 = init_neural_function(input_type, output_type, input_size, output_size, num_units)
        if function2 is None:
            function2 = init_neural_function(input_type, output_type, input_size, output_size, num_units)
        submodules = { "evalfunction" : eval_function, "function1" : function1, "function2" : function2 }
        super().__init__(submodules, input_type, output_type, input_size, output_size, num_units, name=name)

    def execute_on_batch(self, batch, batch_lens=None):
        assert len(batch.size()) == 2

        predicted_eval = self.submodules["evalfunction"].execute_on_batch(batch, batch_lens)
        predicted_function1 = self.submodules["function1"].execute_on_batch(batch, batch_lens)
        predicted_function2 = self.submodules["function2"].execute_on_batch(batch, batch_lens)

        gate = Binarizer2.apply(predicted_eval)        
        assert gate.size() == predicted_function2.size() == predicted_function1.size()
        ite_result = gate*predicted_function1 + (1.0 - gate)*predicted_function2

        return ite_result