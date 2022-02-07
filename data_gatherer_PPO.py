import csv
import os
import subprocess
import sys

import cv2
import numpy as np
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

from map_enemy_id_to_name import id_to_name

sys.path.insert(0, 'evoman')
from gym_environment import Evoman

algorithm = 'PPO'
runs = int(sys.argv[1])
runs_start = int(sys.argv[2])
randomini = sys.argv[3]
if randomini != 'yes' and randomini != 'no':
    raise EnvironmentError()
if runs < 0:
    runs = sys.maxsize

environments = [
    (
        n,
        [(
            Monitor(Evoman(
                enemyn=str(n),
                weight_player_hitpoint=weight_player_hitpoint,
                weight_enemy_hitpoint=1.0 - weight_player_hitpoint,
                randomini=randomini,
            )),
            Monitor(Evoman(
                enemyn=str(n),
                weight_player_hitpoint=1,
                weight_enemy_hitpoint=1,
                randomini=randomini,
            ))
        ) for weight_player_hitpoint in [0.5]]
    )
    for n in [1, 2, 4, 7]
]


class EvalEnvCallback(BaseCallback):
    def __init__(
            self,
            eval_env,
            lengths_path,
            rewards_path,
            models_dir = None,
            video_dir = None,
            raw_data_dir = None,
            verbose: int = 0,
            lengths_prepend: list = [],
            rewards_prepend: list = [],
            n_eval_episodes: int = 5,
            eval_freq: int =  10000,
            model_freq: int = 100000,
            video_freq: int = 25000,
    ):
        super(EvalEnvCallback, self).__init__(verbose=verbose)
        if not os.path.exists(models_dir) and models_dir is not None:
            os.makedirs(models_dir)
        if not os.path.exists(video_dir) and video_dir is not None:
            os.makedirs(video_dir)
        if not os.path.exists(raw_data_dir) and raw_data_dir is not None:
            os.makedirs(raw_data_dir)
        self.eval_env = eval_env
        self.lengths_path = lengths_path
        self.rewards_path = rewards_path
        self.lengths_prepend = lengths_prepend
        self.rewards_prepend = rewards_prepend
        self.video_dir = video_dir
        self.model_freq = model_freq
        self.video_freq = video_freq
        self.n_eval_episodes = n_eval_episodes
        self.eval_freq = eval_freq
        self.lengths = []
        self.rewards = []
        self.models_dir = models_dir
        self.raw_data_dir = raw_data_dir

    def _on_step(self) -> bool:
        if self.n_calls % self.model_freq == 0:
            if self.models_dir is not None:
                self.model.save(f'{self.models_dir}/{self.n_calls}.model')

        if self.n_calls % self.video_freq == 0:
            if self.video_dir is not None:
                obs = self.eval_env.reset()

                fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
                fps = 30
                video_filename = f'{self.video_dir}/{self.n_calls}-temp.avi'
                video_filename_compresed = f'{self.video_dir}/{self.n_calls}.avi'
                out = cv2.VideoWriter(video_filename, fourcc, fps, (self.eval_env.WIDTH, self.eval_env.HEIGHT))
                for _ in range(3500):
                    action, _state = model.predict(obs, deterministic=False)
                    obs, reward, done, info = self.eval_env.step(action)
                    out.write(self.eval_env.render("bgr"))
                    if done:
                        break
                out.release()
                compac = f'ffmpeg -i "{video_filename}" -vcodec h264 "{video_filename_compresed}" -y ; rm "{video_filename}"'
                print(compac)
                os.system(compac)

        if self.n_calls % self.eval_freq == 0:
            self.evaluate()

        return True

    def evaluate(self):
        wins = []
        rs = []
        ls = []
        for j in range(self.n_eval_episodes):
            obs = self.eval_env.reset()
            rew = 0

            for s in range(3500):
                action, _state = self.model.predict(obs, deterministic=False)
                obs, reward, done, info = self.eval_env.step(action)
                rew = rew + reward
                if done:
                    if self.eval_env.env.enemy.life <= 0:
                        wins.append(1)
                    else:
                        wins.append(0)
                    ls.append(s)
                    break
            rs.append(rew)
        self.lengths.append(np.mean(ls))
        self.rewards.append(np.mean(rs))
        with open(f'{self.raw_data_dir}/wins.csv', mode='a') as wins_file:
            wins_writer = csv.writer(wins_file, delimiter=',', quotechar='\'', quoting=csv.QUOTE_NONNUMERIC)
            with open(f'{self.raw_data_dir}/rewards.csv', mode='a') as rewards_file:
                rewards_writer = csv.writer(rewards_file, delimiter=',', quotechar='\'', quoting=csv.QUOTE_NONNUMERIC)
                wins_writer.writerow([self.n_calls, self.n_eval_episodes, ''] + wins)
                rewards_writer.writerow([self.n_calls, self.n_eval_episodes, ''] + rs)


    def _on_training_end(self) -> None:
        self.evaluate()
        with open(f'{self.lengths_path}/Evaluation_lengths.csv', mode='a') as eval_lengths_file:
            l_writer = csv.writer(eval_lengths_file, delimiter=',', quotechar='\'', quoting=csv.QUOTE_NONNUMERIC)
            with open(f'{self.rewards_path}/Evaluation_rewards.csv', mode='a') as eval_rewards_file:
                r_writer = csv.writer(eval_rewards_file, delimiter=',', quotechar='\'', quoting=csv.QUOTE_NONNUMERIC)
                l_writer.writerow(self.lengths_prepend+self.lengths)
                r_writer.writerow(self.rewards_prepend+self.rewards)


for run in range(runs_start, runs_start+runs):
    print(f'Starting run {run}!')
    if randomini:
        baseDir = f'FinalData/RandomIni/{algorithm}/run{run}'
    else:
        baseDir = f'FinalData/StaticIni/{algorithm}/run{run}'

    if not os.path.exists(baseDir):
        os.makedirs(baseDir)

    for enemy_id, enemy_envs in environments:
        enemyDir = f'{baseDir}/{id_to_name(enemy_id)}'
        if not os.path.exists(enemyDir):
            os.makedirs(enemyDir)

        for env, eval_env in enemy_envs:
            modelDir = f'{enemyDir}/models/{({env.env.weight_player_hitpoint}, {env.env.weight_enemy_hitpoint})}'
            videoDir = f'{enemyDir}/videos/{({env.env.weight_player_hitpoint}, {env.env.weight_enemy_hitpoint})}'
            rawDataDir = f'{enemyDir}/raw-data/{({env.env.weight_player_hitpoint}, {env.env.weight_enemy_hitpoint})}'
            if not os.path.exists(modelDir):
                os.makedirs(modelDir)
            if not os.path.exists(videoDir):
                os.makedirs(videoDir)
            if not os.path.exists(rawDataDir):
                os.makedirs(rawDataDir)
            env.env.keep_frames = False
            model = PPO('MlpPolicy', env)
            l_prepend = [f'{id_to_name(enemy_id)}', ""]
            r_prepend = [f'{id_to_name(enemy_id)} ({env.env.weight_player_hitpoint}, {env.env.weight_enemy_hitpoint})', str(env.env.win_value())]
            model.learn(total_timesteps=int(2.5e6), callback=EvalEnvCallback(
                eval_env=eval_env,
                lengths_path=enemyDir,
                rewards_path=enemyDir,
                models_dir=modelDir,
                video_dir=videoDir,
                raw_data_dir=rawDataDir,
                lengths_prepend=l_prepend,
                rewards_prepend=r_prepend,
                eval_freq=12500,
                n_eval_episodes=25,
            ))

            with open(f'{enemyDir}/Training_lengths.csv', mode='a') as train_lengths_file:
                train_lengths_writer = csv.writer(train_lengths_file, delimiter=',', quotechar='\'',
                                                  quoting=csv.QUOTE_NONNUMERIC)
                with open(f'{enemyDir}/Training_rewards.csv', mode='a') as train_rewards_file:
                    train_rewards_writer = csv.writer(train_rewards_file, delimiter=',', quotechar='\'',
                                                      quoting=csv.QUOTE_NONNUMERIC)
                    train_lengths_writer.writerow(l_prepend+env.get_episode_lengths())
                    train_rewards_writer.writerow(r_prepend+env.get_episode_rewards())

                    print(f'\nFinished {id_to_name(enemy_id)} ({env.env.weight_player_hitpoint}, {env.env.weight_enemy_hitpoint})')

        print(f'\n\nFinished {id_to_name(enemy_id)} completely\n\n')