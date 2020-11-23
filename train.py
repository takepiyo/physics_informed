import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import pickle
import time

from Constant import C
from model import Model
from loss_function import gradient_norm_loss

device =  torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def main(data_path, sample_batch_size, boundary_batch_size, max_step):
    dataset = data_load(data_path)
    evaluate_data = get_all_input(dataset)

    model = Model(num_node=56)
    optimizer = optim.SGD(model.parameters(), lr=0.057)
    criterion = nn.MSELoss()
    grad_loss = gradient_norm_loss()

    params = 0
    for p in model.parameters():
        if p.requires_grad:
            params += p.numel()
    print('params: ', params)

    writer = SummaryWriter(log_dir='./logs')

    bc_and_init = define_bc_and_init(dataset)
    start = time.time()
    for step in range(max_step):
        
        optimizer.zero_grad()
        #boundary_and_initial calucrate
        boundary_input, boundary_label = get_boudary_point(bc_and_init, dataset, boundary_batch_size)
        boundary_input, boundary_label = boundary_input.to(device), boundary_label.to(device)
        u = model(boundary_input).to(device)
        loss_1 = criterion(u, boundary_label)
        #sample calucurate
        sample_input, _ = get_sample_point(dataset, sample_batch_size)
        f_values, gradient = f(sample_input, model)
        loss_2 = criterion(f_values, torch.zeros_like(f_values))
        loss_3 = grad_loss(gradient)

        loss = loss_1 + 4.247 * loss_2 + 0.101 * loss_3
        loss.backward()
        optimizer.step()
        if step % 100 == 0:
            print('step {} : loss {} time {}'.format(step, loss, time.time() - start))
            writer.add_scalar('Loss/step',loss, step)
            start = time.time()
        if step % 1000 == 0:
            evaluate(model, evaluate_data, step, writer)
            torch.save(model.state_dict(), 'checkpoint/step_{}.pth'.format(step))
    writer.close()

def get_all_input(dataset):
    '''
    for evaluate
    '''
    input_list = []
    label_list = []
    for t_index, t in enumerate(dataset['t']):
        for x_index, x in enumerate(dataset['x']):
            input_list.append(np.concatenate([t, x], axis=0))
            label_list.append([dataset['u'][x_index][0][t_index]])
    return torch.tensor(np.stack(input_list, axis=0), dtype=torch.float32, device=device), torch.tensor(label_list, dtype=torch.float32, device=device)

def evaluate(model, evaluate_data, step, writer):
    input_data = evaluate_data[0]
    label_data = evaluate_data[1]
    model.eval()
    with torch.no_grad():
        pred = model(input_data)
    L2_loss = torch.sum((pred - label_data) ** 2)
    print('pre_L2loss: {}'.format(L2_loss.item()))
    writer.add_scalar('pre_L2loss/step', L2_loss.item(), step)
    model.train()

def define_bc_and_init(dataset):
    max_time_step = dataset['t'].shape[0]
    left_pos_index = 0
    right_pos_index = dataset['x'].shape[0]
    left_bc = np.concatenate([np.arange(1, max_time_step, dtype=np.int32).reshape(max_time_step - 1, 1), \
                              np.full((max_time_step - 1, 1), left_pos_index)], axis=1)
    #print(left_bc)
    right_bc = np.concatenate([np.arange(1, max_time_step, dtype=np.int32).reshape(max_time_step - 1, 1), \
                              np.full((max_time_step - 1, 1), right_pos_index - 1)], axis=1)
    #print(right_bc)
    initial_con = np.concatenate([np.full((right_pos_index, 1), 0), \
                                  np.arange(right_pos_index, dtype=np.int32).reshape(right_pos_index, 1)], axis=1)
    #print(initial_con)
    #print('*'*80)
    return np.concatenate([left_bc, right_bc, initial_con], axis=0)

def f(sample_input, model):
    u = model(sample_input)
    gradient = torch.autograd.grad(torch.sum(u ,dim=0), sample_input, \
                                   retain_graph=True, create_graph=True)[0]
    f = gradient[:, 0] + C * gradient[:, 1]
    return f, gradient
    
def data_load(data_path):
    with open(data_path,mode='rb') as f:
        data = pickle.load(f)
    return data

def get_label_list(label, batch_point):
    '''
    label [time * stencil]
    batch_point [batch_size * 2(time, point)]
    '''
    label_list = []
    for point in batch_point:
        value = torch.unsqueeze(label[point[0], point[1]], 0)
        label_list.append(value)
    return label_list

def get_sample_point(dataset, sample_batch_size):
    time_step = np.random.randint(1, dataset['t'].shape[0], (sample_batch_size, 1))
    #time_step = np.array([0])
    #time_step = time_step.astype(np.float32)
    sample_stencil = np.random.randint(1, dataset['x'].shape[0] - 2, (sample_batch_size, 1))
    #sample_stencil = sample_stencil.astype(np.float32)
    pair_batch = np.concatenate([time_step, sample_stencil], axis=1)
    #print("pair_batch", pair_batch)
    batch_list = []
    for pair in pair_batch:
        one_dataset = np.concatenate([dataset['t'][pair[0]], \
                                      dataset['x'][pair[1]], \
                                      np.array([dataset['u'][pair[1]][0][pair[0]]])])                             
        batch_list.append(one_dataset)
    numerical_data = np.stack(batch_list, axis=0)
    #print('numerical data', numerical_data)
    return torch.tensor(numerical_data[:, 0:2], dtype=torch.float32, requires_grad=True), torch.tensor(numerical_data[:, 2], dtype=torch.float32).view(-1, 1)

def get_boudary_point(bc_and_init, dataset, boundary_batch_size):

    index_list = np.random.choice(range(bc_and_init.shape[0]), boundary_batch_size)
    pair_batch = np.stack([bc_and_init[index] for index in index_list], axis=0)

    batch_list = []
    for pair in pair_batch:
        one_dataset = np.concatenate([dataset['t'][pair[0]], \
                                      dataset['x'][pair[1]], \
                                      np.array([dataset['u'][pair[1]][0][pair[0]]])])                             
        batch_list.append(one_dataset)

    numerical_data = np.stack(batch_list, axis=0)
    return torch.tensor(numerical_data[:, 0:2], dtype=torch.float32, requires_grad=True), torch.tensor(numerical_data[:, 2], dtype=torch.float32).view(-1, 1)

if __name__ == '__main__':
    
    data_path = 'dataset.pkl'
    sample_batch_size = 5251
    boundary_batch_size = 46
    max_step = 10000000
    main(data_path, sample_batch_size, boundary_batch_size, max_step)

'''
def evaluate(model, dataset, step, writer):
    model.eval()
    L2_loss = 0
    with torch.no_grad():
        for t_index, t in enumerate(dataset['t']):
            for x_index, x in enumerate(dataset['x']):
                #print('t', t.shape)
                #print('x', x)
                input_array = torch.tensor(np.concatenate([t, x]), dtype=torch.float32)
                #print(input_array)
                pred_u = model(input_array)
                L2_loss += (pred_u.item() - dataset['u'][x_index][0][t_index]) ** 2
    writer.add_scalar('pre_L2loss/step', L2_loss.item(), step)
    print('pre_L2loss: {}'.format(L2_loss.item()))
    model.train()

'num_nodes': 56, 'lr': 0.05702160288003877, 'sample_batch_size': 5251, 'boundary_batch_size': 46, 'gradient_loss_weight': 0.10100176756469459, 'sample_loss_weight': 4.247273062146843

def f(sample_input, model):
    f_list = []
    grad_list = []
    for one_sample in sample_input:
        t = one_sample[0].unsqueeze(0)
        x = one_sample[1].unsqueeze(0)
        u = model(torch.cat([t, x], dim=0))
        gradient = torch.autograd.grad(u, (t, x), retain_graph=True, create_graph=True)
        #print('gradient', gradient)
        f = gradient[0] + C * gradient[1]
        f_list.append(f)
        grad_list.extend([gradient[0], gradient[1]])
    return f_list, torch.tensor(grad_list, dtype=torch.float32)
'''