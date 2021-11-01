import cv2
import sys
import threading
import time
from stable_baselines3 import PPO
from stable_baselines3.common.atari_wrappers import MaxAndSkipEnv

sys.path.insert(0, 'evoman')
from gym_environment import Evoman

class EnvTrain (threading.Thread):
    def __init__(self, model, timesteps):
        self.model = model
        self.timesteps = timesteps
        threading.Thread.__init__(self)
    def run(self):
        self.model.learn(self.timesteps)
        print(f'\n\n\nFinished learning {self.name}!\n\n\n')


print(sys.argv)

environments = [MaxAndSkipEnv(Evoman(enemyn=str(a)), skip=2) for a in range(1, 9)]

i = 1

threads = []

for env in environments:
    threads.append(EnvTrain(PPO('MlpPolicy', env, verbose=i), 2**17))
    threads[-1].start()
    while True in [thread.is_alive() for thread in threads]:
        time.sleep(1)
    i = 0

while True in [thread.is_alive() for thread in threads]:
    time.sleep(1)

    # model.set_env(env)
    # model.learn(total_timesteps=(2 ** 16))


    # env.env.keep_frames = True
    # obs = env.reset()

    # fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    # fps = 30
    # video_filename = f'env{i}_({fsn}, 10,2).avi'
    # out = cv2.VideoWriter(video_filename, fourcc, fps, (env.WIDTH, env.HEIGHT))
    # for _ in range(2500):
    #     action, _state = model.predict(obs, deterministic=False)
    #     obs, reward, done, info = env.step(action)
    #     if done:
    #         for frame in env.render('video'):
    #             out.write(frame)
    #         break
    # out.release()