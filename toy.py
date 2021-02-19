import numpy as np
import torch, csv


def bead_spring(data_id, n_sample, dim=2, dt=10**(-3), T_ratio=0.1, m_error=0, rand_seed=0):
    np.random.seed(rand_seed)
    k = 1
    g = 1
    Th = 250
    Tc = Th*T_ratio
    A = np.identity(dim) * (-2*k/g)
    D = np.zeros((dim, dim))
    for i in range(dim):
        if(i+1 < dim):
            A[i, i+1] += k/g
        if(i-1 >= 0):
            A[i, i-1] += k/g
        D[i, i] = Th/g - (Th-Tc)/g/(dim-1)*i
    x0 = np.zeros(dim)
    x = np.zeros((n_sample+1, dim))
    obs = np.zeros((n_sample+1, dim))

    # Relaxation
    for i in range(100000):
        x0 = x0 + dt*np.matmul(x0, A) + np.matmul(np.random.normal(0,1,dim), np.sqrt(2*D*dt))
    x[0, :] = x0
    obs[0, :] = x0 + np.random.normal(0,1,dim)*m_error

    # Generate trajectory data
    for i in range(n_sample):
        x[i+1,:] = x[i,:] + dt*np.matmul(x[i,:], A) + np.matmul(np.random.normal(0,1,dim), np.sqrt(2*D*dt))
        obs[i+1,:] = x[i+1,:] + np.random.normal(0,1,dim)*m_error # Add measurement error
    np.savetxt('Data/data' + str(data_id) + '.txt', obs)

    # Calculate the true entropy production rate
    cov = np.identity(dim)
    for i in range(100000):
        cov += (np.matmul(A, cov) + np.matmul(cov, A.T) + 2*D)*dt
    true_value = np.trace(np.einsum('ij,jk,kl,lm->im', A, np.linalg.inv(D), A, cov)
                        - np.einsum('ij,jk->ik', np.linalg.inv(cov), D))
    print('True entropy production rate : ' + '{:.4f}'.format(true_value))


def Mexican_hat(data_id, n_sample, dt=10**(-4), nonlinear=100, m_error=0, rand_seed=0):
    np.random.seed(rand_seed)
    dim = 2
    k = 1
    g = 1
    Th = 250
    Tc = 25
    A = np.identity(dim) * (-2*k/g)
    D = np.zeros((dim, dim))
    for i in range(dim):
        if(i+1 < dim):
            A[i, i+1] += k/g
        if(i-1 >= 0):
            A[i, i-1] += k/g
        D[i, i] = Th/g - (Th-Tc)/g/(dim-1)*i
    x0 = np.zeros(dim)
    x = np.zeros((n_sample+1, dim))
    obs = np.zeros((n_sample+1, dim))

    # Relaxation
    for i in range(100000):
        x0 = (x0 + dt*np.matmul(x0, A)
              - dt*nonlinear*(4*np.sum(x0**2)-2)*x0 + np.matmul(np.random.normal(0,1,dim), np.sqrt(2*D*dt)))
    x[0, :] = x0  
    obs[0, :] = x0 + np.random.normal(0,1,dim)*m_error

    # Generate trajectory data
    for i in range(n_sample):
        x[i+1,:] = (x[i,:] + dt*np.matmul(x[i,:], A)
                    - dt*nonlinear*(4*np.sum(x[i,:]**2)-2)*x[i,:] + np.matmul(np.random.normal(0,1,dim), np.sqrt(2*D*dt)))
        obs[i+1,:] = x[i+1,:] + np.random.normal(0,1,dim)*m_error # Add measurement error
    np.savetxt('Data/data' + str(data_id) + '.txt', obs)


def breathing_parabola(data_id, n_traj, n_sample, dt=10**(-2), m_error=0, t_error=0, rand_seed=0):
    np.random.seed(rand_seed)
    dim = 1
    time_shift = 0
    x0 = np.zeros(dim)
    x0_tmp = np.zeros(dim)
    x = np.zeros((n_sample+1, dim))
    obs = np.zeros((n_sample+1, dim+1))
    output_file = open('Data/data' + str(data_id) + '.txt', 'w')
    output_writer = csv.writer(output_file, delimiter=' ')

    # Relaxation
    for i in range(10000):
        x0 = x0 - dt*x0 + np.random.normal(0,1,dim) * np.sqrt(2*dt)    
    
    # Generate trajectory data
    for i in range(n_traj):
        # Relaxation
        for j in range(100):
            x0 = x0 - dt*x0 + np.random.normal(0,1,dim) * np.sqrt(2*dt)
        x0_tmp[:] = x0

        # Synchronization error
        if t_error > 0:
            time_shift = np.floor(np.random.uniform(0,1)*t_error/dt)
            for j in range(time_shift):
                x0_tmp = x0_tmp - dt*x0_tmp/(1+j*dt) + np.random.normal(0,1,dim) * np.sqrt(2*dt)
                
        x[0,:] = x0_tmp
        obs[0,0], obs[0,1:] = 0, x0_tmp + np.random.normal(0,1,dim)*m_error
        for j in range(n_sample):
            x[j+1,:] = x[j,:] - dt*x[j,:]/(1+(j+time_shift)*dt) + np.random.normal(0,1,dim) * np.sqrt(2*dt)
            obs[j+1,0], obs[j+1,1:] = (j+1)*dt, x[j+1,:] + np.random.normal(0,1,dim) * m_error

        # Write to the output file 
        output_writer.writerows(obs)
        

def adaptation(data_id, n_traj, n_sample, dt=10**(-5), m_error=0, t_error=0, rand_seed=0):
    np.random.seed(rand_seed)
    dim = 2
    tau_a = 0.02
    tau_m = 0.2
    alpha = 2.7
    A = np.array([[-1/tau_a, (1/tau_a) * alpha],
                  [-1/tau_m, 0]]).transpose()
    Tm = 0.005
    Ta0 = 0.005
    D0 = np.array([[Ta0, 0],
                   [0, Tm]])
    Ta1 = 0.5
    D1 = np.array([[Ta1, 0],
                   [0, Tm]])
    lt = 0.01
    L  = np.array([-lt*dt/tau_a, 0])
    
    x0 = np.zeros(dim)
    x0_tmp = np.zeros(dim)
    x = np.zeros((n_sample+1, dim))
    obs = np.zeros((n_sample+1, dim+1))
    output_file = open('Data/data' + str(data_id) + '.txt', 'w')
    output_writer = csv.writer(output_file, delimiter=' ')

    # Relaxation
    for i in range(100000):
        x0 = x0 + dt*np.matmul(x0, A) + np.matmul(np.random.normal(0,1,dim), np.sqrt(2*D0*dt))

    # Generate trajectory data
    for i in range(n_traj):
        # Relaxation
        for j in range(1000):
            x0 = x0 + dt*np.matmul(x0, A) + np.matmul(np.random.normal(0,1,dim), np.sqrt(2*D0*dt))
        x0_tmp[:] = x0
            
        # Synchronization error
        if t_error > 0:
            time_shift = np.floor(np.random.uniform(0, 1)*t_error/dt)
            for j in range(time_shift):
                x0_tmp = x0_tmp + dt*np.matmul(x0_tmp, A) + L + np.matmul(np.random.normal(0,1,dim), np.sqrt(2*D1*dt))
                
        x[0,:] = x0_tmp
        obs[0,0], obs[0,1:] = 0, x0_tmp + np.random.normal(0,1,dim)*m_error
        for j in range(n_sample):
            x[j+1,:] = x[j,:] + dt*np.matmul(x[j,:], A) + L + np.matmul(np.random.normal(0,1,dim), np.sqrt(2*D1*dt))
            obs[j+1,0], obs[j+1,1:] = (j+1)*dt, x[j+1,:] + np.random.normal(0,1,dim)*m_error

        # Write to the output file
        output_writer.writerows(obs)


def epr_breathing_parabola(t):
    return (t**2 * (3 + 3*t + t**2)**2)/(3 * (1+t)**4 * (3 + 6*t + 6*t**2 + 2*t**3))

                
def epr_adaptation(t):
    return ((np.exp(-50*t) * (-63998940689442 + 6243520070815830*np.exp(50*t) - 171111379578637905*np.exp(100*t)
                             + 24029699524871040*np.exp(150*t) - 1441667880822810*np.exp(200*t)
                             + 44383810651556*np.exp(250*t) - 400*np.exp(25*t)
                             * (192119202 - 13163076234*np.exp(50*t) + 276431059773*np.exp(100*t)
                               - 34658515386*np.exp(150*t) + 842869282*np.exp(200*t))*np.cos(5*np.sqrt(2)*t)
                             + 12.5*(4432189990140 - 625426775264016*np.exp(50*t) + 19698898382037369*np.exp(100*t)
                                     -2423603367706752*np.exp(150*t) + 102783600112652*np.exp(200*t))*np.cos(10*np.sqrt(2)*t)
                             - 4715653140000*np.exp(75*t)*np.cos(15*np.sqrt(2)*t) + 151830378315000*np.exp(125*t)*np.cos(15*np.sqrt(2)*t)
                             - 12837027060000*np.exp(175*t)*np.cos(15*np.sqrt(2)*t) + 1638224935503750*np.exp(50*t)*np.cos(20*np.sqrt(2)*t)
                             - 87205888691521875*np.exp(100*t)*np.cos(20*np.sqrt(2)*t) + 6501171027285000*np.exp(150*t)*np.cos(20*np.sqrt(2)*t)
                             - 42571868625000*np.exp(125*t)*np.cos(25*np.sqrt(2)*t) + 23805931750734375/2.*np.exp(100*t)*np.cos(30*np.sqrt(2)*t)
                             + 845324488800*np.sqrt(2)*np.exp(25*t)*np.sin(5*np.sqrt(2)*t) - 20580429909600*np.sqrt(2)*np.exp(75*t)*np.sin(5*np.sqrt(2)*t)
                             + 521418308281200*np.sqrt(2)*np.exp(125*t)*np.sin(5*np.sqrt(2)*t) - 66720192818400*np.sqrt(2)*np.exp(175*t)*np.sin(5*np.sqrt(2)*t)
                             + 2381271640800*np.sqrt(2)*np.exp(225*t)*np.sin(5*np.sqrt(2)*t) + 16356068262270*np.sqrt(2)*np.sin(10*np.sqrt(2)*t)
                             - 1456450488965340*np.sqrt(2)*np.exp(50*t)*np.sin(10*np.sqrt(2)*t)
                             + 74369430767428335/np.sqrt(2)*np.exp(100*t)*np.sin(10*np.sqrt(2)*t)
                             - 4249583374522860*np.sqrt(2)*np.exp(150*t)*np.sin(10*np.sqrt(2)*t)
                             + 156745750241070*np.sqrt(2)*np.exp(200*t)*np.sin(10*np.sqrt(2)*t)
                             + 4715653140000*np.sqrt(2)*np.exp(75*t)*np.sin(15*np.sqrt(2)*t)
                             - 224367579315000*np.sqrt(2)*np.exp(125*t)*np.sin(15*np.sqrt(2)*t)
                             + 18393897060000*np.sqrt(2)*np.exp(175*t)*np.sin(15*np.sqrt(2)*t)
                             + 647934380057250*np.sqrt(2)*np.exp(50*t)*np.sin(20*np.sqrt(2)*t)
                             - 27461903641202250*np.sqrt(2)*np.exp(100*t)*np.sin(20*np.sqrt(2)*t)
                             + 1894339815860250*np.sqrt(2)*np.exp(150*t)*np.sin(20*np.sqrt(2)*t)
                             + 33640707375000*np.sqrt(2)*np.exp(125*t)*np.sin(25*np.sqrt(2)*t)
                             + 12131083093696875/np.sqrt(2)*np.exp(100*t)*np.sin(30*np.sqrt(2)*t)))
            / (800 * (9801 - 339471*np.exp(50*t) + 26129*np.exp(100*t) + 304425*np.exp(50*t)*np.cos(10*np.sqrt(2)*t)
                      + 24750*np.sqrt(2)*np.exp(50*t)*np.sin(10*np.sqrt(2)*t))**2))

