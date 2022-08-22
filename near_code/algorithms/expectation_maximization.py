import copy
import time
import torch
import numpy as np
from .core import ProgramLearningAlgorithm
from enumeration import ENUMERATION
from program_graph import ProgramGraph
from utils.logging import log_and_print, print_program, print_program_dict
from utils.training import execute_and_train, execute_and_train_set

def probabilize(value_sequence):
    value_sequence = np.array(value_sequence)
    return value_sequence / np.sum(value_sequence)

class EXPECTATIONMAXIMIZATION(ProgramLearningAlgorithm):

    def __init__(self, program_generation_method="fixed"):
        if program_generation_method == "fixed":
            #TODO: un-hardcode this
            self.enumerator = ENUMERATION(max_num_programs=100)
            initial_program_set = self.enumerator.enumerate2depth()
            self.full_program_set = initial_program_set
        else:
            raise NotImplementedError("Non-fixed generation methods for EM not yet implemented!")
    
    def run(self, graph, trainset, validset, train_config, device, verbose=False, train_in_set=True):
        # get the three program architectures that we're going to set
        #TODO: change this hard-coded set once we start sampling!
        if train_in_set:
            trajectory_scores, programs_list = self.train_programs_in_set(graph, trainset, validset, train_config, device, verbose=verbose)
        else:
            trajectory_scores, programs_list = self.train_programs_individually(graph, trainset, validset, train_config, device, verbose=verbose)
        # get minimum loss values from each
        loss_scores = torch.min(torch.stack(trajectory_scores, dim=1), dim=1)
        import pdb; pdb.set_trace()
        # learn the distributional weight over each program in the set
        distribution_values = []
        responsible_programs = loss_scores.indices
        for prog_idx in range(len(programs_list)):
            distribution_values.append(torch.sum(torch.where(responsible_programs == prog_idx, 1, 0), dim=0).item())
        program_distributions = probabilize(distribution_values)
        log_and_print("Overall Minimum Loss for program set is: {}".format(torch.sum(loss_scores.values, dim=0) / len(loss_scores)))
        log_and_print("Program Weight Distribution is: {}".format(program_distributions))
        return programs_list, program_distributions
    
    def train_programs_individually(self, graph, trainset, validset, train_config, device, verbose=False):
        # get the three program architectures that we're going to set
        #TODO: change this hard-coded set once we start sampling!
        program_indices = [-4, -34, -33]
        start_time = time.time()
        trajectory_scores = []
        programs_list = []
        

        for program_idx in program_indices:
            prog_dict = self.full_program_set[program_idx]
            program_candidate = self.full_program_set[program_idx]["program"]
            log_and_print("Training program architecture: {}".format(print_program(program_candidate, ignore_constants=True)))
            overall_score, program, additional_params = execute_and_train(program_candidate, validset, trainset, train_config, 
                graph.output_type, graph.output_size, neural=False, device=device, em_train=True)
            prog_dict["score"] = overall_score
            total_cost = overall_score + prog_dict["struct_cost"]
            prog_dict["path_cost"] = total_cost
            prog_dict["time"] = time.time()-start_time
            programs_list.append(prog_dict)
            log_and_print("Trained program is: \n{}".format(print_program(program, ignore_constants=False)))
            trajectory_scores.append(torch.stack(additional_params["traj_evals"], dim=0))
        return trajectory_scores, programs_list
    
    def train_programs_in_set(self, graph, trainset, validset, train_config, device, verbose=False):
        program_indices = [-4, -34, -33]
        start_time = time.time()
        trajectory_scores = []
        prog_dicts_list = []   
        program_set = []     

        for program_idx in program_indices:
            prog_dict = self.full_program_set[program_idx]
            program_candidate = self.full_program_set[program_idx]["program"]
            # add to set
            prog_dicts_list.append(prog_dict)
            program_set.append(program_candidate)
        # train them jointly
        overall_scores, program_set, additional_params_set = execute_and_train_set(program_set, validset, trainset, train_config, 
                graph.output_type, graph.output_size, neural=False, device=device, em_train=True)
        for idx in range(len(program_set)):
            log_and_print("Trained program is: \n{}".format(print_program(program_set[idx], ignore_constants=False)))
            trajectory_scores.append(torch.stack(additional_params_set[idx]["traj_evals"], dim=0))
            prog_dicts_list[idx]["score"] = overall_scores[idx]
            total_cost = overall_scores[idx] + prog_dicts_list[idx]["struct_cost"]
            prog_dicts_list[idx]["path_cost"] = total_cost
            prog_dicts_list[idx]["time"] = time.time()-start_time
        return trajectory_scores, prog_dicts_list


            
        
 
