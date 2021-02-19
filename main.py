import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import sys, time, copy, os
import param, utils, models


def estimate_entropy(data_id, estimate_id, rand_seed=0):
    '''
    param is fixed as a constant hereafter
    '''
    param.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    torch.manual_seed(rand_seed)  # make results reproducible
    if param.which_model == "FNNt":
        model_func = models.FNNt()
    elif param.which_model == "FNNKt":
        model_func = models.FNNKt()
    elif param.which_model == "FNN":
        model_func = models.FNN()
    model_func.to(param.device)
    optim = torch.optim.Adam(model_func.parameters(), param.alpha)
    data = utils.read_data_file('Data/data' + str(data_id) + '.txt')
    train_data, test_data = np.array_split(data, 2, 1)

    '''
    Calculate basic statistics of the data
    '''
    pos_max = []
    pos_min = []
    pos_mean = []
    pos_std = []
    for i in range(param.dim):
        pos = torch.flatten(data[:, :, i+1])
        pos_max.append(float(torch.max(pos)))
        pos_min.append(float(torch.min(pos)))
        pos_mean.append(float(torch.mean(pos)))
        pos_std.append(float(torch.std(pos)))
    log  = 'pos_max '  + ', '.join(map(str, pos_max))  + '\n'
    log += 'pos_min '  + ', '.join(map(str, pos_min))  + '\n'
    log += 'pos_mean ' + ', '.join(map(str, pos_mean)) + '\n'
    log += 'pos_std '  + ', '.join(map(str, pos_std))  + '\n\n'
    
    '''
    The following process is determined by param.process
    1 : Train the model function
    2 : Estimate the entropy production rate
    3 : Draw the thermodynamic force
    '''        
    if 1 in param.process:
        '''
        Train the model function
        '''
        clock_start = time.time()
        best_score = -10**3  # any large negative value is fine
        best_step  = 0
        best_state = copy.deepcopy(model_func.state_dict())
        f_train = open('Result/train' + str(data_id) + '_' + str(estimate_id) + '.txt', mode='w')
        f_test = open('Result/test' + str(data_id) + '_' + str(estimate_id) + '.txt', mode='w')

        for i in range(param.n_gradient_ascent):
            train_value = utils.train(model_func, train_data, optim)
            test_value = utils.validate(model_func, test_data)
            f_train.write(str(train_value) + ' ')
            f_train.flush()
            f_test.write(str(test_value) + ' ')
            f_test.flush()
            if test_value > best_score:
                best_state = copy.deepcopy(model_func.state_dict())
                best_score = test_value
                best_step  = i+1

        torch.save(best_state, 'Result/model_state' + str(data_id) + '_' + str(estimate_id) + '.txt')
        elapsed_time = time.time() - clock_start
        log += 'time {0}'.format(elapsed_time) + 'sec\n'
        log += 'best_score %f\n' % best_score
        log += 'best_step %d\n\n' % best_step
        f_train.close()
        f_test.close()

        
    if 2 in param.process:
        '''
        Estimate the entropy production rate
        '''
        model_func.load_state_dict(torch.load('Result/model_state' + str(data_id) + '_' + str(estimate_id) + '.txt', map_location=param.device))
        model_func.eval()
        f_epr = open('Result/epr' + str(data_id) + '_' + str(estimate_id) + '.txt', mode='w')
        with torch.no_grad():
            for i in range(len(test_data)):
                epr = utils.estimate_epr(model_func, test_data[i])
                if not param.stationary:
                    f_epr.write("%f " % test_data[i][0][0]) # time instance for the estimation
                f_epr.write("%f " % epr)
                f_epr.flush()
        f_epr.close()

    
    if 3 in param.process:
        '''
        Draw the thermodynamic force
        '''
        model_func.load_state_dict(torch.load('Result/model_state' + str(data_id) + '_' + str(estimate_id) + '.txt', map_location=param.device))
        model_func.eval()
        time_instances = data[:, 0, 0]
        n_time_instances = len(time_instances)
        if param.which_rep == 'TUR':
            const_factors = utils.const_factor(model_func, test_data)
        else:
            const_factors = torch.ones(n_time_instances).to(param.device)

        # Binning the space
        x_max          = pos_mean[param.x_axis] + pos_std[param.x_axis]*3
        x_min          = pos_mean[param.x_axis] - pos_std[param.x_axis]*3
        y_max          = pos_mean[param.y_axis] + pos_std[param.y_axis]*3
        y_min          = pos_mean[param.y_axis] - pos_std[param.y_axis]*3
        bin_width_x    = (x_max - x_min)/param.n_bin
        bin_width_y    = (y_max - y_min)/param.n_bin
        x_centers      = (np.arange(param.n_bin) + 0.5) * bin_width_x + x_min
        y_centers      = (np.arange(param.n_bin) + 0.5) * bin_width_y + y_min
        x_mesh, y_mesh = np.meshgrid(x_centers, y_centers)
        fx             = np.zeros([param.n_bin, param.n_bin])
        fy             = np.zeros([param.n_bin, param.n_bin])
        f_abs          = np.zeros([param.n_bin, param.n_bin])
        
        if param.stationary:
            state = torch.tensor(pos_mean)
            fig_format = '.svg'
        else:
            state = torch.tensor([0] + pos_mean)
            fig_format = '.png'
        per = max(n_time_instances//param.n_figure_max, 1)
        
        with torch.no_grad():
            for i in range(n_time_instances):
                if i % per != 0:
                    continue
                for jx in range(param.n_bin):
                    for jy in range(param.n_bin):
                        if param.stationary:
                            state[param.x_axis] = x_min + (jx + 0.5) * bin_width_x
                            state[param.y_axis] = y_min + (jy + 0.5) * bin_width_y
                        else:
                            state[0] = time_instances[i]
                            state[param.x_axis+1] = x_min + (jx + 0.5) * bin_width_x
                            state[param.y_axis+1] = y_min + (jy + 0.5) * bin_width_y
                        force = model_func(state, const_factors[i])
                        fx[jy, jx] = float(force[param.x_axis])
                        fy[jy, jx] = float(force[param.y_axis])
                        f_abs[jy, jx] = float(force[param.x_axis]**2 + force[param.y_axis]**2) ** 0.5

                # Draw instantaneous thermodynamic force
                if i==0:
                    quiverkey_size = f_abs.mean() # The reference vector size is determined on the basis of the initial thermodynamic force
                fig, ax = plt.subplots()
                q = ax.quiver(x_mesh, y_mesh, fx, fy, f_abs, width = 0.006)
                ax.quiverkey(q, X=0, Y=1.06, U=quiverkey_size, label='Quiver key, length=' + '{:.3f}'.format(quiverkey_size), labelpos='E')
                if not param.stationary:
                    ax.set_title('t=' + '{:.3f}'.format(i*param.slice_interval*param.dt), fontsize=18)
                plt.tick_params(labelsize=18)
                fig.savefig('Result/thermo_force' + str(data_id) + '_' + str(estimate_id) + '_' + str(i) + fig_format)
                if n_time_instances > 1: 
                    plt.close()  # Unless it is closed, every plots will appear in Jupyter notebook.
                
        # Make a gif movie
        images = []
        if n_time_instances > 1:
            for i in range(n_time_instances):
                if i % per != 0:
                    continue
                im = Image.open('Result/thermo_force' + str(data_id) + '_' + str(estimate_id) + '_' + str(i) + fig_format)
                images.append(im)
                os.remove('Result/thermo_force' + str(data_id) + '_' + str(estimate_id) + '_' + str(i) + fig_format)
            images[0].save('Result/thermo_force' + str(data_id) + '_' + str(estimate_id) + '.gif', save_all = True,
                           append_images=images[1:], loop=0, duration=3000//n_time_instances, quality=100)
        else:
            os.rename('Result/thermo_force' + str(data_id) + '_' + str(estimate_id) + '_' + str(i) + fig_format,
                      'Result/thermo_force' + str(data_id) + '_' + str(estimate_id) + fig_format)

    '''
    Write log in the end of this function
    '''
    f_log = open('Result/log' + str(data_id) + '_' + str(estimate_id) + '.txt', mode='w')
    f_log.write(log)
    f_log.flush()
    f_log.close()

    

# Main Program
if __name__ == '__main__':
    '''
    data_id and estimate_id are identification numbers.
    Input:  'Data/data{data_id}.txt'
    Output: 'Result/log{data_id}_{estimate_id}.txt', 'Result/train{data_id}_{estimate_id}.txt',
            'Result/test{data_id}_{estimate_id}.txt', 'Result/epr{data_id}_{estimate_id}.txt',
            'Result/model_state{data_id}_{estimate_id}.txt', 'Result/thermo_force{data_id}_{estimate_id}.gif',

    The other parameters (param.name) can also be set here by param.name = float(sys.argv[3])
    See param.py for their explanation.
    '''
    data_id      = int(sys.argv[1])
    estimate_id  = int(sys.argv[2])
    estimate_entropy(data_id, estimate_id)
