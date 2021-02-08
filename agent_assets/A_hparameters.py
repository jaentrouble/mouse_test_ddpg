Buffer_size = 200000
Learn_start = 20000
Batch_size = 32
Target_update = 100
Target_update_tau = 1e-1
Q_discount = 0.99
Train_epoch = 1

class Lr():
    def __init__(self):
        self.start = None
        self.end = None
        self.halt_steps = None
        self.nsteps = None
        self.epsilon = None
        self.grad_clip = None

lr = {
    'actor' : Lr(),
    'critic' : Lr(),
    'encoder' : Lr(),
}
lr['actor'].halt_steps = 0
lr['actor'].start = 0.001
lr['actor'].end = 0.00005
lr['actor'].nsteps = 2000000
lr['actor'].epsilon = 1e-2
lr['actor'].grad_clip = 1.0

lr['critic'].halt_steps = 0
lr['critic'].start = 0.001
lr['critic'].end = 0.00005
lr['critic'].nsteps = 2000000
lr['critic'].epsilon = 1e-2
lr['critic'].grad_clip = 1.0

lr['encoder'].halt_steps = 0
lr['encoder'].start = 1e-4
lr['encoder'].end = 1e-5
lr['encoder'].nsteps = 1e6
lr['encoder'].epsilon = 1e-2
lr['encoder'].grad_clip = 1.0

lr['forward'] = lr['encoder']
lr['inverse'] = lr['encoder']

OUP_damping = 0.15
OUP_stddev_start=0.2
OUP_stddev_end = 0.05
OUP_stddev_nstep = 500000
OUP_clip = 0.8

IQN_ENABLE = True
IQN_SUPPORT = 64
IQN_COS_EMBED = 64

ICM_ENABLE = True
ICM_intrinsic = 1.0

class _Buf():
    def __init__(self):
        self.alpha = 0.6
        self.beta = 0.4
        self.epsilon = 1e-3
        self.N = 5

Buf = _Buf()

Model_save = 200000

histogram = 100000
log_per_steps = 100

