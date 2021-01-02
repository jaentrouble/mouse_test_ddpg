from gym.envs.registration import register

register(
    id='mouseClCont-v0',
    entry_point='gym_mouse.envs:MouseEnv_cl'
)

