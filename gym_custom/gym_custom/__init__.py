from gym.envs.registration import register
register(id='QPendulum-v0',
        entry_point='gym_custom.envs:QPendulumEnv',
        max_episode_steps=200,
)
register(id='ImPendulum-v0',
        entry_point='gym_custom.envs:ImPendulumEnv',
        max_episode_steps=200,
)
register(id='DMPendulum-v0',
        entry_point='gym_custom.envs:PendulumDmEnv',
        max_episode_steps=200,
)
register(id='QAcrobot-v0',
        entry_point='gym_custom.envs:QAcrobotEnv',
)
