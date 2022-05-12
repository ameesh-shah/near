import copy
from distutils.log import error
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

class GANDiscriminator(nn.Module):

    def __init__(self, input_size, num_units, num_layers=1):
        super(GANDiscriminator, self).__init__()
        self.input_size = input_size
        self.output_size = 1
        self.rnn_size = num_units
        self.num_layers = num_layers
        self.rnn = nn.LSTM(self.input_size, self.rnn_size, num_layers=self.num_layers).to(device)
        self.out_layer = nn.Linear(self.rnn_size, self.output_size).to(device)
        self.out_activation = nn.Sigmoid()

    def init_hidden(self, batch_size):
        ahid = torch.zeros(self.num_layers, batch_size, self.rnn_size)
        bhid = torch.zeros(self.num_layers, batch_size, self.rnn_size)
        ahid = ahid.requires_grad_(True)
        bhid = bhid.requires_grad_(True)
        hid = (ahid.to(device), bhid.to(device))
        return hid

    def forward(self, batch, batch_lens):
        assert isinstance(batch, torch.Tensor)
        batch_size, seq_len, feature_dim = batch.size()

        # pass through rnn
        hidden = self.init_hidden(batch_size)
        batch_packed = torch.nn.utils.rnn.pack_padded_sequence(batch, batch_lens, batch_first=True, enforce_sorted=False)
        self.rnn.flatten_parameters()
        out, hidden = self.rnn(batch_packed, hidden)
        out, _ = torch.nn.utils.rnn.pad_packed_sequence(out, batch_first=True)

        # pass through linear layer
        out = out.contiguous()
        out = out.view(-1, out.shape[2])
        out = self.out_activation(self.out_layer(out))
        out = out.view(batch_size, seq_len, -1)

        return out


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

def process_generator_batch(program, env, batch_size, device='cpu'):
    #TODO: vectorize this implementation, if possible
    trajectories = []
    # generate trajectories, using the env.
    for idx in range(batch_size):
        current_state = env.reset()
        current_traj = [current_state]
        done = False
        while not done:
            action = program.execute_on_single(torch.Tensor(current_state).to(device))
            current_state, rew, done, info = env.step(action.cpu().detach().numpy())
            current_traj.append(current_state)
            # TODO: check if this is a bug or just not checked for
            if len(current_traj) > 500:
                break
        trajectories.append(current_traj)
    # evaluate the collected trajectories
    return trajectories 

def process_batch(program, batch, output_type, output_size, device='cpu'):
    if len(torch.tensor(batch[0]).size()) > 1:
        # we're in the list input setting
        batch_input = [torch.tensor(traj) for traj in batch]
        batch_padded, batch_lens = pad_minibatch(batch_input, num_features=batch_input[0].size(1))
        batch_padded = batch_padded.to(device)
        batch_padded, batch_lens = pad_minibatch(batch_input, num_features=batch_input[0].size(1))
        batch_padded = batch_padded.to(device)
        out_padded = program.execute_on_batch(batch_padded, batch_lens)
        out_unpadded = unpad_minibatch(out_padded, batch_lens, listtoatom=(program.output_type=='atom'))
        if output_size == 1 or output_type == "list":
            return flatten_tensor(out_unpadded).squeeze()
        else:
            if isinstance(out_unpadded, list):
                out_unpadded = torch.cat(out_unpadded, dim=0).to(device)          
            return out_unpadded
    else:
        # in the atom setting - no padding needed
        batch_input = torch.tensor(batch)
        batch_in = torch.tensor(batch_input).float().to(device)
        batch_out = program.execute_on_batch(batch_in, None)
        # if output_size == 1 or output_type == "list":
        #     return flatten_tensor(batch_out).squeeze()
        # else:
        if isinstance(batch_out, list):
            batch_out = torch.cat(batch_out, dim=0).to(device)          
        return batch_out

def process_discriminator_batch(discriminator_net, trajectory_batch):
    batch_input = [torch.tensor(traj) for traj in trajectory_batch]
    batch_padded, batch_lens = pad_minibatch(batch_input, num_features=batch_input[0].size(1))
    batch_padded = batch_padded.to(device)
    batch_padded, batch_lens = pad_minibatch(batch_input, num_features=batch_input[0].size(1))
    batch_padded = batch_padded.to(device)
    # get discriminator output
    d_out = discriminator_net.forward(batch_padded, batch_lens)
    idx = torch.tensor(batch_lens).to(device) - 1
    idx = idx.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, d_out.size(-1))
    model_out = d_out.gather(1, idx).squeeze(1)
    return model_out.squeeze().to(device)

def execute_and_train_generator(program, groundtruthset, env, train_config, output_size,
    neural=False, device='cpu', print_every=60):
    lr = train_config['lr']
    neural_epochs = train_config['neural_epochs']
    symbolic_epochs = train_config['symbolic_epochs']
    optimizer = train_config['optimizer']
    batch_size = train_config['batch_size']
    discrim_units = train_config['num_discriminator_units']
    lossfxn = nn.BCELoss()

    discriminator = GANDiscriminator(program.input_size, discrim_units)

    generator_optim = init_optimizer(program, optimizer, lr)
    discriminator_optim = optimizer(discriminator.parameters(), lr=lr)
    
    real_label = 1.0
    fake_label = 0.0

    best_program = None
    best_metric = float('inf')
    best_additional_params = {}

    num_epochs = neural_epochs if neural else symbolic_epochs
    for epoch in range(1, num_epochs + 1):
        for batchidx in range(len(groundtruthset)):
            gt_input = groundtruthset[batchidx]
            ## TRAIN THE DISCRIMINATOR 
            ################################
            # first, train with all-GT batch
            discriminator_optim.zero_grad()
            # pass through discriminator
            gt_outputs = process_discriminator_batch(discriminator, gt_input)
            labels = torch.full((batch_size,), real_label, dtype=torch.float).to(device)
            error_gt = lossfxn(gt_outputs, labels)
            # backprop on gt error
            error_gt.backward()
            D_x = gt_outputs.mean().item()

            # now, train with the generated batch
            gen_input = process_generator_batch(program, env, batch_size, device)
            labels.fill_(fake_label)
            gen_outputs = process_discriminator_batch(discriminator, gen_input)
            error_gen = lossfxn(gen_outputs, labels)
            error_gen.backward()
            D_G_z1 = gen_outputs.mean().item()
            # error of D = sum of error over GT and Gen batches.
            error_discrim = error_gt + error_gen
            discriminator_optim.step()

            ## TRAIN THE GENERATOR 
            ################################
            #TODO: hacky solution to no-tunable parameter program
            if generator_optim is not None:
                generator_optim.zero_grad()
                labels.fill_(real_label)
                # perform the discriminator forward pass again
                gen_outputs = process_discriminator_batch(discriminator, gen_input)
                error_gnrator = lossfxn(gen_outputs, labels)
                error_gnrator.backward()
                D_G_z2 = gen_outputs.mean().item()
                generator_optim.step()
                print_gnr_error = error_gnrator.item()
                print_DGz2_error = D_G_z2
            else:
                print_gnr_error = -999
                print_DGz2_error = -999
            if batchidx % print_every == 1:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                    % (epoch, num_epochs, batchidx, len(groundtruthset),
                        error_discrim.item(), print_gnr_error, D_x, D_G_z1, print_DGz2_error))


        best_program = copy.deepcopy(program)
        best_metric = print_gnr_error
        best_additional_params = {"discriminator_error" : error_discrim, "D_x": D_x,
                                  "D_G_z1" : D_G_z1, "D_G_z2" : print_DGz2_error}
        # select model with best validation score
    program = copy.deepcopy(best_program)
    log_and_print("Program generator error is: {:.4f}".format(best_metric))
    log_and_print("Program discriminator error is: {:.4f}".format(
        best_additional_params["discriminator error"]
    ))

    return best_metric



def execute_and_train(program, validset, trainset, train_config, output_type, output_size, 
    neural=False, device='cpu', use_valid_score=False, print_every=60):

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
    validation_true_vals = torch.tensor(flatten_batch(validation_output, is_classification=is_classification)).float().to(device)
    # TODO a little hacky, but easiest solution for now
    if isinstance(lossfxn, nn.CrossEntropyLoss):
        validation_true_vals = validation_true_vals.long()

    best_program = None
    best_metric = float('inf')
    best_additional_params = {}

    for epoch in range(1, num_epochs+1):
        for batchidx in range(len(trainset)):
            batch_input, batch_output = map(list, zip(*trainset[batchidx]))
            true_vals = torch.tensor(flatten_batch(batch_output, is_classification=is_classification)).float().to(device)
            predicted_vals = process_batch(program, batch_input, output_type, output_size, device)
            # TODO a little hacky, but easiest solution for now
            if isinstance(lossfxn, nn.CrossEntropyLoss):
                true_vals = true_vals.long()
            #print(predicted_vals.shape, true_vals.shape)
            loss = lossfxn(predicted_vals, true_vals)
            if curr_optim is not None:
                #TODO: hacky solution to dealing with parameter-less programs
                curr_optim.zero_grad()
                loss.backward()
                curr_optim.step()


            # if batchidx % print_every == 0 or batchidx == 0:
            #     log_and_print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch, num_epochs, loss.item()))

        # check score on validation set
        with torch.no_grad():
            predicted_vals = process_batch(program, validation_input, output_type, output_size, device)
            if is_classification:
                metric, additional_params = evalfxn(predicted_vals, validation_true_vals, num_labels=num_labels)
            else:
                metric = evalfxn(predicted_vals, validation_true_vals)
                additional_params = None

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
    
    return best_metric
