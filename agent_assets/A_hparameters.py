Buffer_size = 200000
Learn_start = 20000
Batch_size =32
Target_update = 100
Target_update_tau = 1e-1
# epsilon = 0.1
# epsilon_min = 0.001
# epsilon_nstep = 500000
Q_discount = 0.99
Train_epoch = 1
lr_start = 0.001
lr_end = 0.00005
lr_nsteps = 2000000

OUP_damping = 0.15
OUP_stddev_start=0.2
OUP_stddev_end = 0.05
OUP_stddev_nstep = 500000

class Buf():
    alpha = 0.6
    beta = 0.4
    epsilon = 1e-3

Model_save = 200000

histogram = 100000
log_per_steps = 100

