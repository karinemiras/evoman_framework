import sys

import numpy as np
from stable_baselines3 import PPO, DQN
sys.path.insert(0, 'evoman')
from gym_environment import Evoman
from gym_environment_discrete import Evoman as EvomanDiscrete

env = Evoman()
env2 = EvomanDiscrete()
model_ppo = PPO('MlpPolicy', env)
model_dqn = DQN('MlpPolicy', env2)

print(model_ppo)
print(model_dqn)