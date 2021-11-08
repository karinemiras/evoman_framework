import cv2
import sys
import csv
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.atari_wrappers import MaxAndSkipEnv
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv

sys.path.insert(0, 'evoman')
from gym_environment import Evoman

environments = [VecFrameStack(DummyVecEnv([lambda: MaxAndSkipEnv(Monitor(Evoman(enemyn=str(n))), skip=2)]), n_stack=3) for n in range(1, 9)]

i = 0

with open(f'Generalist/data_gather_lengths.csv', mode='a') as lengths_file:
    lengths_writer = csv.writer(lengths_file, delimiter=',', quotechar='\'', quoting=csv.QUOTE_NONNUMERIC)
    with open(f'Generalist/data_gather_rewards.csv', mode='a') as rewards_file:
        rewards_writer = csv.writer(rewards_file, delimiter=',', quotechar='\'', quoting=csv.QUOTE_NONNUMERIC)

        model = PPO('MlpPolicy', environments[0], verbose=1)
        lengths = []
        rewards = []
        for k in range(len(environments)):
            lengths.append([])
            lengths[k].extend([f'env{k+1}', ''])
            rewards.append([])
            rewards[k].extend([f'env{k+1}', ''])

        for time in range(18, 10, -1):
            i = 0
            for env in environments:
                model.set_env(env)
                model.learn(total_timesteps=(2 ** time))
                lengths[i].extend(env.envs[0].get_episode_lengths())
                rewards[i].extend(env.envs[0].get_episode_rewards())
                i += 1


        i = 0
        for env in environments:
            lengths_writer.writerow(lengths[i])
            rewards_writer.writerow(rewards[i])
            i += 1
            for j in range(10):
                obs = env.reset()

                fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
                fps = 15
                video_filename = f'Generalist/env{i}_run{j}.avi'
                out = cv2.VideoWriter(video_filename, fourcc, fps, (env.envs[0].WIDTH, env.envs[0].HEIGHT))
                for _ in range(2500):
                    action, _state = model.predict(obs, deterministic=False)
                    obs, reward, done, info = env.step(action)
                    out.write(env.envs[0].render("bgr"))
                    if done:
                        break
                out.release()


# for env in environments:
#     i += 1
#
#     model.set_env(env)
#     model.learn(total_timesteps=(2 ** 17))
#
#     print(f'\n\n\nFinished learning env{i}!\n\n\n')
#
#     obs = env.reset()
#
#     fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
#     fps = 30
#     video_filename = f'env{i}_({fsn}, 10,2).avi'
#     out = cv2.VideoWriter(video_filename, fourcc, fps, (env.WIDTH, env.HEIGHT))
#     for _ in range(2500):
#         action, _state = model.predict(obs, deterministic=False)
#         obs, reward, done, info = env.step(action)
#         out.write(env.render("bgr"))
#         if done:
#             break
#     out.release()