from turtle import position
import torch
import torch.nn as nn
import numpy as np
from .neural_functions import init_neural_function
from .library_functions import AffineFeatureSelectionFunction, AffineFunction, LibraryFunction, SimpleITE
import os

ANT_MODELS = []
for direction in ['up', 'down', 'left', 'right']:
    filename = os.getcwd() + '/prl_code/primitives/ant/' + direction + '.pt'
    ANT_MODELS.append(torch.load(filename))

INTERSECTION_FEATURE_SUBSETS = {
    "position": torch.LongTensor([0, 1]),
    "goal_pos": torch.LongTensor([113, 114]),
    "all_pos": torch.LongTensor([0, 1, 113, 114]),
    "primitive_features": torch.LongTensor(range(2,113))
}
INTERSECTION_FULL_SIZE = 115


# TODO allow user to choose device
if torch.cuda.is_available():
    device = 'cuda:0'
else:
    device = 'cpu'


class AntAffineFunction(LibraryFunction):

    def __init__(self, raw_input_size, selected_input_size, output_size, num_units, name="Affine"):
        self.selected_input_size = selected_input_size
        super().__init__({}, "atom", "singleatom", raw_input_size,
                         output_size, num_units, name=name, has_params=True)

    def init_params(self):
        self.linear_layer = nn.Linear(
            self.selected_input_size, self.output_size, bias=True).to(device)
        self.parameters = {
            "weights": self.linear_layer.weight,
            "bias": self.linear_layer.bias
        }

    def execute_on_batch(self, batch, batch_lens=None):
        assert len(batch.size()) == 2
        return self.linear_layer(batch)


class AntAffineFeatureSelectionFunction(AntAffineFunction):

    def __init__(self, input_size, output_size, num_units, name="AffineFeatureSelection"):
        assert hasattr(self, "full_feature_dim")
        assert input_size >= self.full_feature_dim
        if self.full_feature_dim == 0:
            self.is_full = True
            self.full_feature_dim = input_size
        else:
            self.is_full = False
        additional_inputs = input_size - self.full_feature_dim

        assert hasattr(self, "feature_tensor")
        assert len(self.feature_tensor) <= input_size
        self.feature_tensor = self.feature_tensor.to(device)
        super().__init__(raw_input_size=input_size, selected_input_size=self.feature_tensor.size()[-1]+additional_inputs,
                         output_size=output_size, num_units=num_units, name=name)

    def init_params(self):
        self.raw_input_size = self.input_size
        if self.is_full:
            self.full_feature_dim = self.input_size
            self.feature_tensor = torch.arange(self.input_size).to(device)

        additional_inputs = self.raw_input_size - self.full_feature_dim
        self.selected_input_size = self.feature_tensor.size(
        )[-1] + additional_inputs
        self.linear_layer = nn.Linear(
            self.selected_input_size, self.output_size, bias=True).to(device)
        self.parameters = {
            "weights": self.linear_layer.weight,
            "bias": self.linear_layer.bias
        }

    def execute_on_batch(self, batch, batch_lens=None):
        assert len(batch.size()) == 2
        features = torch.index_select(batch, 1, self.feature_tensor)
        remaining_features = batch[:, self.full_feature_dim:]
        return self.linear_layer(torch.cat([features, remaining_features], dim=-1))

class AntPositionSelection(AntAffineFeatureSelectionFunction):

    def __init__(self, input_size, output_size, num_units):
        self.full_feature_dim = INTERSECTION_FULL_SIZE
        self.feature_tensor = INTERSECTION_FEATURE_SUBSETS["all_pos"]
        super().__init__(input_size, output_size, num_units, name="PositionSelect")

class AntGoalPosSelection(AntAffineFeatureSelectionFunction):

    def __init__(self, input_size, output_size, num_units):
        self.full_feature_dim = INTERSECTION_FULL_SIZE
        self.feature_tensor = INTERSECTION_FEATURE_SUBSETS["all_pos"]
        super().__init__(input_size, output_size, num_units, name="GoalPositionSelect")


class AntBehaviorPrimitiveMovement(LibraryFunction):

    def __init__(self, input_size, output_size, num_units, primitive_direction, name):
        self.behavior_primitive = ANT_MODELS[primitive_direction]
        submodules = {}
        self.obs_subset = INTERSECTION_FEATURE_SUBSETS["primitive_features"].to(device)
        self.behavior_primitive.to(device)
        super().__init__(submodules, "atom", "atom", input_size, output_size, num_units, name=name, has_params=False)
    
    def execute_on_batch(self, batch, batch_lens=None):
        with torch.no_grad():
            new_batch = torch.index_select(batch, 1, self.obs_subset).to(device)
            return self.behavior_primitive.act(new_batch, deterministic=True, on_device=True)

class AntUpPrimitiveFunction(AntBehaviorPrimitiveMovement):

    def __init__(self, input_size, output_size, num_units):
        super().__init__(input_size, output_size, num_units, 0, "AntUpPrimitive")

class AntDownPrimitiveFunction(AntBehaviorPrimitiveMovement):

    def __init__(self, input_size, output_size, num_units):
        super().__init__(input_size, output_size, num_units, 1, "AntDownPrimitive")

class AntLeftPrimitiveFunction(AntBehaviorPrimitiveMovement):

    def __init__(self, input_size, output_size, num_units):
        super().__init__(input_size, output_size, num_units, 2, "AntLeftPrimitive")

class AntRightPrimitiveFunction(AntBehaviorPrimitiveMovement):

    def __init__(self, input_size, output_size, num_units):
        super().__init__(input_size, output_size, num_units, 3, "AntRightPrimitive")
