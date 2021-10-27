import cv2
import sys
import csv
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv

sys.path.insert(0, 'evoman')
from gym_environment import Evoman

environments = [VecFrameStack(DummyVecEnv([lambda: Monitor(Evoman(enemyn='1'))]), n_stack=fsn) for fsn in range(1, 7)]

i = 0

with open(f'FSN/FSN_VFS_data_episode_lengths.csv', mode='w') as lengths_file:
    lengths_writer = csv.writer(lengths_file, delimiter=',', quotechar='\'', quoting=csv.QUOTE_NONNUMERIC)
    with open(f'FSN/FSN_VFS_data_episode_rewards.csv', mode='w') as rewards_file:
        rewards_writer = csv.writer(rewards_file, delimiter=',', quotechar='\'', quoting=csv.QUOTE_NONNUMERIC)

        for env in environments:
            i += 1
            model = PPO('MlpPolicy', env, verbose=1)
            model.learn(total_timesteps=(2 ** 20))
            lengths = env.envs[0].get_episode_lengths()
            lengths.reverse()
            lengths.append('')
            lengths.append(i)
            lengths.reverse()
            rewards = env.envs[0].get_episode_rewards()
            rewards.reverse()
            rewards.append('')
            rewards.append(i)
            rewards.reverse()

            lengths_writer.writerow(lengths)
            rewards_writer.writerow(rewards)


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