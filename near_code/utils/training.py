import copy
from distutils.log import error
from re import L
from regex import R
import torch
import torch.nn as nn
import numpy as np
import dsl


from utils.data import pad_minibatch, unpad_minibatch, flatten_tensor, flatten_batch
from utils.logging import log_and_print

# TODO allow user to choose device
if torch.cuda.is_available():
    device = 'cuda:0'
else:
    device = 'cpu'

def init_optimizer(program, optimizer, lr):
    queue = [program]
    all_params = []
    while len(queue) != 0:
        current_function = queue.pop()
        if issubclass(type(current_function), dsl.HeuristicNeuralFunction):
            current_function.init_model()
            all_params.append({'params' : current_function.model.parameters(),'lr' : lr})
        elif current_function.has_params:
            current_function.init_params()
            all_params.append({'params': list(current_function.parameters.values()), 'lr': lr})
        else:
            for submodule, functionclass in current_function.submodules.items():
                queue.append(functionclass)
    if len(all_params) == 0:
        return None
    curr_optim = optimizer(all_params, lr)
    return curr_optim

def process_batch(program, batch, output_type, output_size, device='cpu', is_em=False):
    if len(torch.tensor(batch[0]).size()) > 1:
        # we're in the list input setting
        batch_input = [torch.tensor(traj) for traj in batch]
        batch_padded, batch_lens = pad_minibatch(batch_input, num_features=batch_input[0].size(1))
        batch_padded = batch_padded.to(device)
        out_padded = program.execute_on_batch(batch_padded, batch_lens)
        out_unpadded = unpad_minibatch(out_padded, batch_lens, listtoatom=(program.output_type=='atom'))
        if output_size == 1 or output_type == "list":
            return flatten_tensor(out_unpadded).squeeze(), batch_lens
        else:
            if isinstance(out_unpadded, list):
                out_unpadded = torch.cat(out_unpadded, dim=0).to(device)          
            return out_unpadded, batch_lens
    else:
        # in the atom setting - no padding needed
        batch_input = torch.tensor(batch)
        batch_in = torch.tensor(batch_input).float().to(device)
        batch_out = program.execute_on_batch(batch_in, None)
        if isinstance(batch_out, list):
            batch_out = torch.cat(batch_out, dim=0).to(device)     
        return batch_out, None

def execute_and_train_set(programset, validset, trainset, train_config, output_type, output_size, 
    neural=False, device='cpu', use_valid_score=False, em_train=False, print_every=60):
    # TODO: consolidate this with execute_and_train, if possible.
    lr = train_config['lr']
    neural_epochs = train_config['neural_epochs']
    symbolic_epochs = train_config['symbolic_epochs']
    optimizer = train_config['optimizer']
    lossfxn = train_config['lossfxn']
    evalfxn = train_config['evalfxn']
    num_labels = train_config['num_labels']
    is_classification = train_config['is_classification']

    num_epochs = neural_epochs if neural else symbolic_epochs

    # initialize an optimizer for each program
    optim_set = []
    for program in programset:
        optim_set.append(init_optimizer(program, optimizer, lr))

    # prepare validation set
    validation_input, validation_output = map(list, zip(*validset))
    #import pdb; pdb.set_trace()
    validation_true_vals = torch.tensor(flatten_batch(validation_output, is_classification=is_classification, is_em=em_train)).float().to(device)
    # TODO a little hacky, but easiest solution for now
    if isinstance(lossfxn, nn.CrossEntropyLoss):
        validation_true_vals = validation_true_vals.long()

    best_program_set = [None for _ in range(len(programset))]
    best_metric_set = [float('inf') for _ in range(len(programset))]
    best_additional_params_set = [{} for _ in range(len(programset))]

    for epoch in range(1, num_epochs+1):
        for batchidx in range(len(trainset)):
            batch_input, batch_output = map(list, zip(*trainset[batchidx]))
            true_vals = torch.tensor(flatten_batch(batch_output, is_classification=is_classification, is_em=em_train)).float().to(device)
            program_losses = []
            for program in programset:
                # first, collect the losses for each program
                predicted_vals, batch_lens = process_batch(program, batch_input, output_type, output_size, device)
                # TODO a little hacky, but easiest solution for now
                if isinstance(lossfxn, nn.CrossEntropyLoss):
                    true_vals = true_vals.long()
                #print(predicted_vals.shape, true_vals.shape)
                # train a set of programs using expectation-maximization
                if output_type == "list":   
                    len_idx = 0
                    losses = []
                    for traj_len in batch_lens:
                        # TODO: speed this up by taking slices?
                        if traj_len == 0:
                            continue
                        # calculate normalized losses per trajectory
                        # calculate normalized losses per *program*
                        losses.append(lossfxn(predicted_vals[len_idx:len_idx+traj_len], true_vals[len_idx:len_idx+traj_len]) / traj_len)
                        len_idx += traj_len
                        # softmax the losses
                    loss_tensor = torch.stack(losses)
                    # weight the trajectories against one another
                    # TODO: do we need trajectory-weighting? (the commented out lines)
                    # exp_loss_vals = 1 / torch.exp(loss_tensor.detach())
                    # batch_softmax_vals = torch.softmax(exp_loss_vals, dim=0)
                    # loss = torch.sum(loss_tensor * batch_softmax_vals, dim=0)
                    program_losses.append(loss_tensor)
                else:     
                    #TODO: implement the atom case (should be a simplification of the list case!)   
                    loss_tensor = torch.stack([lossfxn(predicted_vals[idx], true_vals[idx]) for idx in predicted_vals.shape[0]])
                    program_losses.append(loss_tensor)
            all_program_loss_tensor = torch.stack(program_losses).detach()
            exp_loss_vals = 1 / torch.exp(all_program_loss_tensor)
            program_softmax_vals = torch.softmax(exp_loss_vals, dim=0)
            summed_program_losses = []
            # do this iteratively in order to keep the losses separate
            for loss_val_idx in range(len(program_losses)):
                summed_program_losses.append(torch.sum(program_losses[loss_val_idx] * program_softmax_vals[loss_val_idx], dim=0))
            # softmax the program losses depending on how well they each did in comparison
            for program_idx in range(len(programset)):
                curr_optim = optim_set[program_idx]
                loss_val = summed_program_losses[program_idx]
                if curr_optim is not None:
                    #TODO: hacky solution to dealing with parameter-less programs
                    curr_optim.zero_grad()
                    loss_val.backward()
                    curr_optim.step()


            if batchidx % print_every == 0 or batchidx == 0:
                log_and_print('Epoch [{}/{}]:'.format(epoch, num_epochs))
                for progidx, loss_val in enumerate(summed_program_losses):
                    log_and_print('\tLoss for program {} : {:.4f}'.format(progidx, loss_val.item()))

        # check score on validation set
        with torch.no_grad():
            for program_idx in range(len(programset)):
                program = programset[program_idx]
                predicted_vals, batch_lens = process_batch(program, validation_input, output_type, output_size, device)
                if is_classification:
                    
                    metric, additional_params = evalfxn(predicted_vals, validation_true_vals, num_labels=num_labels)
                else:
                    metric = evalfxn(predicted_vals, validation_true_vals)
                    additional_params = {}
                    
                    # in expectation-maximization
                    if output_type == "list":   
                        len_idx = 0
                        evals = []
                        for traj_len in batch_lens:
                            # TODO: speed this up by taking slices?
                            if traj_len == 0:
                                continue
                            # calculate normalized losses per trajectory
                            evals.append(evalfxn(predicted_vals[len_idx:len_idx+traj_len], true_vals[len_idx:len_idx+traj_len]) / traj_len)
                            len_idx += traj_len
                        additional_params["traj_evals"] = evals 

                if use_valid_score:
                    if sum(metric) < sum(best_metric_set[program_idx]):
                        best_program_set[program_idx] = copy.deepcopy(program)
                        best_metric_set[program_idx] = metric
                        best_additional_params_set[program_idx] = additional_params
                else:
                    best_program_set[program_idx] = copy.deepcopy(program)
                    best_metric_set[program_idx] = metric
                    best_additional_params_set[program_idx] = additional_params

    # select model with best validation score
    programset = copy.deepcopy(best_program_set)
    for progidx in range(len(programset)):
        log_and_print("EVALUATION FOR PROGRAM {}:".format(progidx + 1))
        if is_classification:
                log_and_print("Validation score is: {:.4f}".format(best_metric_set[progidx]))
                log_and_print("Average f1-score is: {:.4f}".format(1 - best_metric_set[progidx]))
                log_and_print("Hamming accuracy is: {:.4f}".format(best_additional_params_set[progidx]['hamming_accuracy']))
        else:
            log_and_print("Validation score is: {}".format(best_metric_set))
    if em_train:
        return best_metric_set, best_program_set, best_additional_params_set
    
    return best_metric_set



def execute_and_train(program, validset, trainset, train_config, output_type, output_size, 
    neural=False, device='cpu', use_valid_score=False, em_train=False, print_every=60):

    lr = train_config['lr']
    neural_epochs = train_config['neural_epochs']
    symbolic_epochs = train_config['symbolic_epochs']
    optimizer = train_config['optimizer']
    lossfxn = train_config['lossfxn']
    evalfxn = train_config['evalfxn']
    num_labels = train_config['num_labels']
    is_classification = train_config['is_classification']

    num_epochs = neural_epochs if neural else symbolic_epochs

    # initialize optimizer
    curr_optim = init_optimizer(program, optimizer, lr)

    # prepare validation set
    validation_input, validation_output = map(list, zip(*validset))
    #import pdb; pdb.set_trace()
    validation_true_vals = torch.tensor(flatten_batch(validation_output, is_classification=is_classification, is_em=em_train)).float().to(device)
    # TODO a little hacky, but easiest solution for now
    if isinstance(lossfxn, nn.CrossEntropyLoss):
        validation_true_vals = validation_true_vals.long()

    best_program = None
    best_metric = float('inf')
    best_additional_params = {}

    for epoch in range(1, num_epochs+1):
        for batchidx in range(len(trainset)):
            batch_input, batch_output = map(list, zip(*trainset[batchidx]))
            true_vals = torch.tensor(flatten_batch(batch_output, is_classification=is_classification, is_em=em_train)).float().to(device)
            predicted_vals, batch_lens = process_batch(program, batch_input, output_type, output_size, device)
            # TODO a little hacky, but easiest solution for now
            if isinstance(lossfxn, nn.CrossEntropyLoss):
                true_vals = true_vals.long()
            #print(predicted_vals.shape, true_vals.shape)
            if em_train:
                # in expectation-maximization
                if output_type == "list":   
                    len_idx = 0
                    losses = []
                    for traj_len in batch_lens:
                        # TODO: speed this up by taking slices?
                        if traj_len == 0:
                            continue
                        # calculate normalized losses per trajectory
                        losses.append(lossfxn(predicted_vals[len_idx:len_idx+traj_len], true_vals[len_idx:len_idx+traj_len]) / traj_len)
                        len_idx += traj_len
                        # softmax the losses
                    loss_tensor = torch.stack(losses)
                    exp_loss_vals = 1 / torch.exp(loss_tensor.detach())
                    batch_softmax_vals = torch.softmax(exp_loss_vals, dim=0)
                    #loss = torch.sum(loss_tensor, dim=0)
                    loss = torch.sum(loss_tensor * batch_softmax_vals, dim=0)
            else:        
                loss = lossfxn(predicted_vals, true_vals)
            if curr_optim is not None:
                #TODO: hacky solution to dealing with parameter-less programs
                curr_optim.zero_grad()
                loss.backward()
                curr_optim.step()


            if batchidx % print_every == 0 or batchidx == 0:
                log_and_print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch, num_epochs, loss.item()))

        # check score on validation set
        with torch.no_grad():
            predicted_vals, batch_lens = process_batch(program, validation_input, output_type, output_size, device)
            if is_classification:
                
                metric, additional_params = evalfxn(predicted_vals, validation_true_vals, num_labels=num_labels)
            else:
                metric = evalfxn(predicted_vals, validation_true_vals)
                additional_params = {}
                if em_train:
                    # in expectation-maximization
                    if output_type == "list":   
                        len_idx = 0
                        evals = []
                        for traj_len in batch_lens:
                            # TODO: speed this up by taking slices?
                            if traj_len == 0:
                                continue
                            # calculate normalized losses per trajectory
                            evals.append(evalfxn(predicted_vals[len_idx:len_idx+traj_len], true_vals[len_idx:len_idx+traj_len]) / traj_len)
                            len_idx += traj_len
                        additional_params["traj_evals"] = evals 

        if use_valid_score:
            if metric < best_metric:
                best_program = copy.deepcopy(program)
                best_metric = metric
                best_additional_params = additional_params
        else:
            best_program = copy.deepcopy(program)
            best_metric = metric
            best_additional_params = additional_params

    # select model with best validation score
    program = copy.deepcopy(best_program)
    if is_classification:
        log_and_print("Validation score is: {:.4f}".format(best_metric))
        log_and_print("Average f1-score is: {:.4f}".format(1 - best_metric))
        log_and_print("Hamming accuracy is: {:.4f}".format(best_additional_params['hamming_accuracy']))
    else:
        log_and_print("Validation score is: {:.4f}".format(best_metric))
    if em_train:
        return best_metric, best_program, best_additional_params
    
    return best_metric
