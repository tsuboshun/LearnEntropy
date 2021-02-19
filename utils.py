import torch
import numpy as np
import matplotlib.pyplot as plt
import random, time
import param, utils, models


def estimate_epr(model_func, data_inst):
    '''
    data_inst is an ensemble of single-step transitions at a particular time of interest
    '''
    if param.stationary:
        # For the stationary case, s = (x) 
        s  = data_inst[:, :param.dim]
        dx = data_inst[:, param.dim:]
    else:
        # For the non-stationary case, s = (t x) 
        s  = data_inst[:, :param.dim+1]
        dx = data_inst[:, param.dim+1:]
        
    cur = (model_func(s) * dx).sum(1)
    if param.which_estimator == "Simple":
        return (2*cur.mean()-cur.var()/2) / (param.dt*param.current_interval)
    elif param.which_estimator == "NEEP":
        return (cur-torch.exp(-cur)+1).mean() / (param.dt*param.current_interval)
    elif param.which_estimator == "TUR":
        return 2*cur.mean()**2/cur.var() / (param.dt*param.current_interval)
    elif param.which_estimator == "Var":
        return cur.var()/2 / (param.dt*param.current_interval)
        
    
def obj_func(model_func, data, coeff):
    if param.stationary:
        # For the stationary case, s = (x) 
        s  = data[:, :, :param.dim]
        dx = data[:, :, param.dim:]
    else:
        # For the non-stationary case, s = (t x) 
        s  = data[:, :, :param.dim+1]
        dx = data[:, :, param.dim+1:]

    cur = (model_func(s) * dx).sum(2)
    if param.which_rep == "Simple":
        return torch.matmul(coeff, 2*torch.mean(cur, 1) - torch.var(cur, 1)/2) / (param.dt*param.current_interval)
    elif param.which_rep == "NEEP":
        return torch.matmul(coeff, torch.mean(cur-torch.exp(-cur)+1, 1)) / (param.dt*param.current_interval)
    elif param.which_rep == "TUR":
        return torch.matmul(coeff, 2*torch.mean(cur, 1)**2/torch.var(cur, 1)) / (param.dt*param.current_interval)


def const_factor(model_func, data):
    if param.stationary:
        # For the stationary case, s = (x) 
        s  = data[:, :, :param.dim]
        dx = data[:, :, param.dim:]
    else:
        # For the non-stationary case, s = (t x) 
        s  = data[:, :, :param.dim+1]
        dx = data[:, :, param.dim+1:]

    cur = (model_func(s) * dx).sum(2)
    return torch.var(cur, 1)/torch.mean(cur, 1)/2
    
    
def train(model_func, train_data, optim):
    '''
    This function performs the gradient ascent using training data,
    and returns the value of the objective function.
    '''
    coeff = np.random.uniform(0, 1, len(train_data)).reshape(1, -1)
    coeff /= coeff.sum()
    coeff = torch.from_numpy(coeff).to(param.device).float()
    
    model_func.train()
    loss = -obj_func(model_func, train_data, coeff)
    optim.zero_grad()
    loss.backward()
    optim.step()
    return float(-loss)


def validate(model_func, data):
    '''
    This function evaluates the model function using training or test data,
    and returns an estimate of the time-averaged entropy production rate.
    '''
    coeff = (np.ones(len(data))/len(data)).reshape(1, -1)
    coeff = torch.from_numpy(coeff).to(param.device).float()
    model_func.eval()
    return float(obj_func(model_func, data, coeff))


def read_data_file(filename):
    # The following way of reading a file is much faster than np.loadtxt(filename).
    raw_data = " ".join(open(filename).readlines()).split()
    raw_data = np.array(raw_data, dtype=float).reshape(param.n_traj, -1)
    
    if param.stationary:
        mid_pos = (raw_data[:, :-param.dim*param.current_interval] + raw_data[:, param.dim*param.current_interval:])/2 
        mid_pos = mid_pos.reshape(param.n_traj, -1, param.dim)
        disp = raw_data[:, param.dim*param.current_interval:] - raw_data[:, :-param.dim*param.current_interval]
        disp = disp.reshape(param.n_traj, -1, param.dim)
        data = np.concatenate([mid_pos, disp], 2).reshape(1, -1, param.dim*2)  # data.shape = (1, n_transition, dim*2)
    else:
        mid_pos = (raw_data[:, :-(param.dim+1)*param.current_interval] + raw_data[:, (param.dim+1)*param.current_interval:])/2 
        mid_pos = mid_pos.reshape(param.n_traj, -1, param.dim+1)[:, ::param.slice_interval, :]
        disp = raw_data[:, (param.dim+1)*param.current_interval:] - raw_data[:, :-(param.dim+1)*param.current_interval]
        disp = disp.reshape(param.n_traj, -1, param.dim+1)[:, ::param.slice_interval, 1:]
        data = np.einsum('jik', np.concatenate([mid_pos, disp], 2))  # data.shape = (n_time_instance, n_traj, dim*2+1)
    return torch.from_numpy(data).to(param.device).float()
        

def plot_trajectory(filename):
    raw_data = " ".join(open(filename).readlines()).split()
    raw_data = np.array(raw_data, dtype=float).reshape(param.n_traj, -1)[0, :] # Draw the first trajectory
    
    if param.stationary:
        data = raw_data.reshape(-1, param.dim)
    else:
        data = raw_data.reshape(-1, param.dim+1)[:,1:]
    x = data[:, param.x_axis]
    y = data[:, param.y_axis]
    n = len(x)
    k = 10
    # We interpolate x and y with k intermediary points.
    x2 = np.interp(np.arange(n*k), np.arange(n)*k, x)
    y2 = np.interp(np.arange(n*k), np.arange(n)*k, y)

    fig, ax = plt.subplots()
    ax.scatter(x2, y2, c=range(n*k), linewidth=0,
               marker='o', s=2, cmap=plt.cm.winter)
    plt.tick_params(labelsize=18)
    plt.savefig('Result/trajectory.png', dpi=200)


def plot_learning_curve(filename_train, filename_test, true_value=-1):
    train_value = " ".join(open(filename_train).readlines()).split()
    train_value = np.array(train_value, dtype=float)
    test_value = " ".join(open(filename_test).readlines()).split()
    test_value = np.array(test_value, dtype=float)
    step = np.arange(len(train_value))+1
    
    fig, ax = plt.subplots()
    ax.plot(step, train_value, 'limegreen', linewidth=3, label='train')
    ax.plot(step, test_value, 'royalblue', linewidth=3, label='test')
    if true_value >= 0:
        ax.hlines([true_value], 0, len(train_value)+1, 'gray', linestyles='dashed', linewidth=2)
    plt.legend(fontsize=14)
    plt.tick_params(labelsize=18)
    plt.savefig('Result/learning_curve.svg')

    
def plot_epr(filename, true_epr_func=None):
    time_epr = " ".join(open(filename).readlines()).split()
    time_epr = np.array(time_epr, dtype=float)
    time_instances = time_epr[::2]
    epr = time_epr[1::2]

    fig, ax = plt.subplots()
    ax.plot(time_instances, epr, color='royalblue', marker='.', linestyle='None', markersize=8)

    if true_epr_func != None:
        time = np.linspace(param.t_init, param.t_fin, 1000)
        ax.plot(time, true_epr_func(time), color='black', linewidth=2)

    plt.tick_params(labelsize=18)
    plt.savefig('Result/epr.svg')

    

