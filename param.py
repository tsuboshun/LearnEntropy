'''
*** Process of the program ***
1 : Train the model function
2 : Estimate the entropy production rate
3 : Draw the thermodynamic force
''' 
process = [1, 2, 3]


'''
*** Information on the trajectories *** 
stationary : True or False
dim        : dimension of the system
n_traj     : number of trajectories
dt         : sampling interval
t_init     : initial time (non-stationary case)
t_fin      : final time (non-stationary case)
'''
stationary = True
dim    = 2
n_traj = 1
dt     = 10**(-3)
t_init = 0
t_fin  = 1


'''
*** Setting of the model function and the algorithm ***
which_model       : FNNt, FNNKt, or FNN (name with letter t is for the non-stationary case)
which_rep         : Simple, NEEP, or TUR
which_estimator   : Simple, NEEP, TUR, or Var
n_layer           : number of hidden layers
n_hidden          : number of units per hidden layer
n_output          : number of units in the output layer (for FNNKt)
current_inteval   : time interval of the optimizing current in the unit of dt
slice_inteval     : This program will include time instances every slice_interval for calculating the objective function,
                    i.e., pairs of positions at time {(0, current_interval*dt), (slice_interval*dt, [slice_interval+current_interval]*dt), ...}
                    will be used for calculating the objective function.  
n_gardient_ascent : iteration number of the gradient ascent
alpha             : step size of the gradient ascent
'''
which_model = "FNN"
which_rep   = "TUR"
which_estimator = "TUR"
n_layer     = 3
n_hidden    = 30
n_output    = 20
current_interval = 1
slice_interval   = 1
n_gradient_ascent = 100
alpha = 10**(-3)


'''
*** Setting of the thermodynamic force drawing ***
n_bin        : number of binning in each axis. n_bin * n_bin vectors will be drawn.
x_axis       : It specifies which dimension to use as the x-axis (x_axis = 0, 1, ..., or dim-1).
y_axis       : It specifies which dimension to use as the y-axis (y_axis = 0, 1, ..., or dim-1).
n_figure_max : maximum number of figures to make a gif movie

The x_axis and y_axis are also applied to utils.plot_trajectory()
'''
n_bin = 12
x_axis = 0
y_axis = 1
n_figure_max = 100
