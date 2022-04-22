import logging
import os
from queue import Empty
import dsl


def init_logging(save_path):
    logfile = os.path.join(save_path, 'log.txt')

    # clear log file
    with open(logfile, 'w'):
        pass
    # remove previous handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(filename=logfile, level=logging.INFO)

def log_and_print(line):
    print(line)
    logging.info(line)

def print_program(program, ignore_constants=False):
    if not isinstance(program, dsl.LibraryFunction):
        return program.name
    else:
        collected_names = []
        for submodule, functionclass in program.submodules.items():
            collected_names.append(print_program(functionclass, ignore_constants=ignore_constants))
        if program.has_params:
            parameters = "params: {}".format(program.parameters.values())
            if not ignore_constants:
                collected_names.append(parameters)
        joined_names = ', '.join(collected_names)
        return program.name + "(" + joined_names + ")"

def print_program_dict(prog_dict):
    log_and_print(print_program(prog_dict["program"], ignore_constants=True))
    log_and_print("struct_cost {:.4f} | score {:.4f} | path_cost {:.4f} | time {:.4f}".format(
        prog_dict["struct_cost"], prog_dict["score"], prog_dict["path_cost"], prog_dict["time"]))

def bring_to_cpu(program):
    # given a program, send the parameters back to the cpu for ease at inference time
    q = [program]
    while len(q) > 0: # while it's not empty, bfs
        curr_mod = q.pop()
        for submodule, functionclass in curr_mod.submodules.items():
            # add submodules to the queue
            q.append(functionclass)
        if curr_mod.has_params:
            breakpoint()
            # bring parameters back to cpu
            for param_name, param in curr_mod.parameters.items():
                curr_mod.parameters[param_name] = param.to('cpu')
    breakpoint()