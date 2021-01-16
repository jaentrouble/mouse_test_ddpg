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

class Lr():
    def __init__(self):
        self.start = None
        self.end = None
        self.nsteps = None

lr = {
    'actor' : Lr(),
    'critic' : Lr(),
}
lr['actor'].start = 0.001
lr['actor'].end = 0.00005
lr['actor'].nsteps = 2000000

lr['critic'].start = 0.001
lr['critic'].end = 0.00005
lr['critic'].nsteps = 2000000


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

