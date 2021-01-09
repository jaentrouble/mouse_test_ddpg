from gym.envs.registration import register

register(
    id='mouseUnity-v0',
    entry_point='gym_mouse.envs:MouseEnv_unity'
)

