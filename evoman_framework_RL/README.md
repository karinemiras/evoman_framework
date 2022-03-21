Evoman is a video game playing framework inspired on Megaman.

A demo can be found here:  https://www.youtube.com/watch?v=ZqaMjd1E4ZI

# Reinforcement Learning Extension
This branch of the EvoMan environment introduces two OpenAI Gym Environment adaptation of the original EvoMan Environment.
The adaptations are identical save for the input.
`gym_environment` takes 5 binary inputs, which `gym_environment_discrete` takes a single input between 0 and 31.
The discrete input is then split into 5 binary inputs. This is introduced to deal with the limitations of some Reinforcement Learning Algorithms.
For this discrete input, it is important to realise that actions which are similar number-wise are not necessarily similar when it comes to the performed move.

A description of OpenAI's Gym environments can be found at https://gym.openai.com/.
A library which uses OpenAI Gym environments can be found at https://github.com/DLR-RM/stable-baselines3.