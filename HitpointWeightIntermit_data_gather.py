import csv
import sys

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.atari_wrappers import MaxAndSkipEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv

from map_enemy_id_to_name import id_to_name

sys.path.insert(0, 'evoman')
from gym_environment import Evoman

environments = [
    [(
        VecFrameStack(DummyVecEnv(
            [lambda: MaxAndSkipEnv(Monitor(Evoman(
                enemyn=str(n),
                weight_player_hitpoint=weight_player_hitpoint / 10.0,
                weight_enemy_hitpoint=1.0 - (weight_player_hitpoint / 10.0),
                randomini=True,
            )), skip=2)]
        ), n_stack=3),
        VecFrameStack(DummyVecEnv(
            [lambda: MaxAndSkipEnv(Monitor(Evoman(
                enemyn=str(n),
                weight_player_hitpoint=1,
                weight_enemy_hitpoint=1,
                randomini=True,
            )), skip=2)]
        ), n_stack=3)
    ) for weight_player_hitpoint in range(11)]
    for n in range(1, 9)
]


class EvalEnvCallback(BaseCallback):
    def __init__(
            self,
            eval_env,
            l_writer,
            r_writer,
            verbose: int = 0,
            lengths_prepend: list = [],
            rewards_prepend: list = [],
            n_eval_episodes: int = 5,
            eval_freq: int = 10000,
    ):
        super(EvalEnvCallback, self).__init__(verbose=verbose)
        self.eval_env = eval_env
        self.l_writer = l_writer
        self.r_writer = r_writer
        self.lengths_prepend = lengths_prepend
        self.rewards_prepend = rewards_prepend
        self.n_eval_episodes = n_eval_episodes
        self.eval_freq = eval_freq
        self.lengths = []
        self.rewards = []

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            rs = []
            ls = []
            for j in range(self.n_eval_episodes):
                obs = self.eval_env.reset()
                rs.append(0)

                for s in range(3500):
                    action, _state = self.model.predict(obs, deterministic=False)
                    obs, reward, done, info = self.eval_env.step(action)
                    rs[j] = rs[j] + reward
                    if done:
                        ls.append(s)
                        break
            self.lengths.append(np.mean(ls))
            self.rewards.append(np.mean(rs))

        return True

    def _on_training_end(self) -> None:
        self.l_writer.writerow(self.lengths_prepend+self.lengths)
        self.r_writer.writerow(self.rewards_prepend+self.rewards)



for enemy_id, enemy_envs in enumerate(environments, start=1):
    with open(f'HitpointWeightIntermit/{id_to_name(enemy_id)}_lengths.csv', mode='a') as lengths_file:
        lengths_writer = csv.writer(lengths_file, delimiter=',', quotechar='\'', quoting=csv.QUOTE_NONNUMERIC)
        with open(f'HitpointWeightIntermit/{id_to_name(enemy_id)}_rewards.csv', mode='a') as rewards_file:
            rewards_writer = csv.writer(rewards_file, delimiter=',', quotechar='\'', quoting=csv.QUOTE_NONNUMERIC)

            for env, eval_env in enemy_envs:
                env.envs[0].env.env.keep_frames = False
                model = PPO('MlpPolicy', env)
                model.learn(total_timesteps=(2 ** 18), callback=EvalEnvCallback(
                    eval_env=eval_env,
                    l_writer=lengths_writer,
                    r_writer=rewards_writer,
                    lengths_prepend=[f'{id_to_name(enemy_id)}', ""],
                    rewards_prepend=[f'{id_to_name(enemy_id)} ({env.envs[0].env.env.weight_player_hitpoint}, {env.envs[0].env.env.weight_enemy_hitpoint})', str(env.envs[0].env.env.win_value())],
                    n_eval_episodes=15,
                    eval_freq=10000,
                ))
                # lengths = env.envs[0].get_episode_lengths()
                # lengths.reverse()
                # lengths.append(np.mean(env.envs[0].get_episode_lengths()))
                # lengths.append()
                # lengths.reverse()
                # rewards = env.envs[0].get_episode_rewards()
                # rewards.reverse()
                # rewards.append()
                # rewards.append()
                # rewards.reverse()

                # lengths_writer.writerow(lengths)
                # rewards_writer.writerow(rewards)
                print(f'\nFinished {id_to_name(enemy_id)} ({env.envs[0].env.env.weight_player_hitpoint}, {env.envs[0].env.env.weight_enemy_hitpoint})')

    print(f'\n\nFinished {id_to_name(enemy_id)} completely\n\n')
                # env.envs[0].env.env.keep_frames = True
                # for j in range(10):
                #     obs = env.reset()
                #
                #     fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
                #     fps = 30
                #     video_filename = f'Optimized_FStack_FSkip/PPO_env{i}_run{j}.avi'
                #     out = cv2.VideoWriter(video_filename, fourcc, fps, (env.envs[0].WIDTH, env.envs[0].HEIGHT))
                #     for _ in range(2500):
                #         action, _state = model.predict(obs, deterministic=False)
                #         obs, reward, done, info = env.step(action)
                #         if done:
                #             break
                #     for frame in env.render("p_video"):
                #         out.write(frame)
                #     out.release()


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