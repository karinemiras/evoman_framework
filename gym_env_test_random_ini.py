import cv2
import sys
from stable_baselines3 import PPO

sys.path.insert(0, 'evoman')
from gym_environment import Evoman

environments = [Evoman(enemyn=str(a), randomini=True) for a in range(1, 9)]

model = PPO('MlpPolicy', environments[1], verbose=1)

i = 0

for env in environments:
    i += 1

    model.set_env(env)
    model.learn(total_timesteps=(2 ** 20))

    print("\n\n\nFinished learning env1!\n\n\n")

    obs = env.reset()

    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    fps = 30
    video_filename = f'env{i}_randomini.avi'
    out = cv2.VideoWriter(video_filename, fourcc, fps, (env.WIDTH, env.HEIGHT))
    for _ in range(2500):
        action, _state = model.predict(obs, deterministic=False)
        obs, reward, done, info = env.step(action)
        out.write(env.render("bgr"))
        if done:
            break
    out.release()