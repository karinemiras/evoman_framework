import cv2
import sys
from stable_baselines3 import PPO

sys.path.insert(0, 'evoman')
from gym_environment import Evoman

env = Evoman(enemyn='2', randomini=False, cost_per_timestep=0.02)

model = PPO('MlpPolicy', env, verbose=1)

# for i in range(100):
model.learn(total_timesteps=2**22)
obs = env.reset()
for i in range(100):
    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    fps = 30
    video_filename = f'fast_is_better_randomini_long_{i}.avi'
    out = cv2.VideoWriter(video_filename, fourcc, fps, (env.WIDTH, env.HEIGHT))

    for _ in range(2500):
        action, _state = model.predict(obs, deterministic=False)
        obs, reward, done, info = env.step(action)
        out.write(env.render("bgr"))
        if done:
            break
    out.release()