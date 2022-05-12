from turtle import position
import torch
import torch.nn as nn
import numpy as np
from .neural_functions import init_neural_function
from .library_functions import AffineFeatureSelectionFunction, AffineFunction, LibraryFunction, SimpleITE, AntSimpleITE
import os

ANT_MODELS = []
for direction in ['up', 'down', 'left', 'right']:
    filename = os.getcwd() + '/prl_code/primitives/ant/' + direction + '.pt'
    ANT_MODELS.append(torch.load(filename))

# behaviors in ant models: up, down, left, right

INTERSECTION_FEATURE_SUBSETS = {
    "position": torch.LongTensor([0, 1]),
    "all_goal_pos": torch.LongTensor([113, 114, 115, 116, 117, 118]),
    "all_pos": torch.LongTensor([0, 1, 113, 114, 113, 114, 115, 116, 117, 118]),
    "left_goal_pos": torch.LongTensor([113, 114]),
    "up_goal_pos": torch.LongTensor([115, 116]),
    "right_goal_pos": torch.LongTensor([117, 118]),
    "primitive_features": torch.LongTensor(range(2,113))
}
LEFT_GOAL = torch.LongTensor([[6, -6]])
UP_GOAL = torch.LongTensor([[12, 0]])
RIGHT_GOAL = torch.LongTensor([[6, 6]])
ALL_GOALS = [LEFT_GOAL, UP_GOAL, RIGHT_GOAL]
INTERSECTION_FULL_SIZE = 119
# goals (in order): left, up, right

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
    
    def execute_on_single(self, state):
        return self.linear_layer(state)


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

    def execute_on_single(self, state):
        features = torch.index_select(state, 0, self.feature_tensor)
        remaining_features = state[:, self.full_feature_dim:]
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
    
    def execute_on_single(self, state):
        with torch.no_grad():
            new_ex = torch.index_select(state, 0, self.obs_subset).to(device)
            return self.behavior_primitive.act(new_ex, deterministic=True, on_device=True)

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

class MoveAntToClosestGoal(LibraryFunction):

    def __init__(self, input_size, output_size, num_units):
        submodules = {}
        self.obs_subset = INTERSECTION_FEATURE_SUBSETS["primitive_features"].to(device)
        self.position_subset = INTERSECTION_FEATURE_SUBSETS["position"]
        self.left_goal = INTERSECTION_FEATURE_SUBSETS["left_goal_pos"].to(device)
        self.up_goal = INTERSECTION_FEATURE_SUBSETS["up_goal_pos"].to(device)
        self.right_goal = INTERSECTION_FEATURE_SUBSETS["right_goal_pos"].to(device)

        super().__init__(submodules, "atom", "atom", input_size, output_size, num_units, name="AntToClosestGoal", has_params=False)
    
    def get_closest_goal(self, inp, is_batch=False):
        # find the minimum goal
        # compute distance from minimum
        with torch.no_grad():
            axis = 1 if is_batch else 0
            current_pos = torch.index_select(inp, axis, self.obs_subset).to(device)
            closest_goal_idx =  np.argmin([nn.functional.pairwise_distance(goal, current_pos).item() for goal in ALL_GOALS])
            return ALL_GOALS[closest_goal_idx]

    
    def execute_on_batch(self, batch, batch_lens=None):
        with torch.no_grad():
            new_batch = torch.index_select(batch, 1, self.obs_subset).to(device)
            return self.behavior_primitive.act(new_batch, deterministic=True, on_device=True)
    
    def execute_on_single(self, state):
        with torch.no_grad():
            current_goal = self.get_closest_goal(state)
            curr_pos = torch.index_select(state, 0, self.position_subset).to(device)
            if current_goal == UP_GOAL:
                behavior_primitive = ANT_MODELS[0].to(device)
            else:
                if torch.abs(curr_pos[0] - current_goal[0]) > 1.0:  # if it's not in line on the y-axis
                    behavior_primitive = ANT_MODELS[0] if curr_pos[0] < current_goal[0] else ANT_MODELS[1]
                else:  # it's in line, so move it towards the goal
                    behavior_primitive = ANT_MODELS[2] if current_goal == LEFT_GOAL else ANT_MODELS[3]
                behavior_primitive = behavior_primitive.to(device)
            new_ex = torch.index_select(state, 0, self.obs_subset).to(device)
            return behavior_primitive.act(new_ex, deterministic=True, on_device=True)

