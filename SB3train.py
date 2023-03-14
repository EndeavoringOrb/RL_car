from stable_baselines3 import PPO, DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

from custom_env import racingEnv
import pygame
import numpy as np

"""models_dir = f"SB3models/{int(time.time())}/"

if not os.path.exists(models_dir):
	os.makedirs(models_dir)"""

print("making env...")
img_num = 4 #int(input("Enter the image number: "))
steps = 2048
my_image = pygame.image.load(f'image{img_num}.png')
np_img = np.load(f"image{img_num}.npy")
config = np.load(f"config{img_num}.npy", allow_pickle=True)
env = racingEnv(np_img, my_image, config)
env = DummyVecEnv([lambda: env])
model_number = int(input("Enter the model number: "))
batch_size = steps//32
load_model = input("Load model? [Y/n]: ")
if load_model.lower() == "y":
	model_path = input("Enter relative model path: ")
	weird_char =  [char for char in model_path if char not in "abcdefghijklmnopqrstuvwxyz1234567890_."][0]
	loaded_timesteps = int(model_path.split(weird_char)[1].split(".")[0])
if load_model.lower() == "y":
	#model = PPO('MlpPolicy', env, verbose=2, n_steps=steps, batch_size=batch_size)
	model = PPO.load(model_path,env=env)
else:
	model_params = {
		"policy": "MlpPolicy",
		"learning_rate": 1e-3,
		"buffer_size": 10000,
		"learning_starts": 0,
		"batch_size": 32,
		"tau": 1.0,
		"gamma": 0.99,
		"train_freq": 4,
		"gradient_steps": 1,
		"target_update_interval": 1000,
		"exploration_fraction": 0.1,
		"exploration_initial_eps": 1.0,
		"exploration_final_eps": 0.02,
	}
	model = DQN(env=env, **model_params)
	loaded_timesteps = 0
	#model = PPO('MlpPolicy', env, verbose=2, n_steps=steps, batch_size=batch_size)

iters = 0
while iters < 1000000000000:
	print(f"learning step: {iters}")
	iters += 1
	model.learn(total_timesteps=steps, reset_num_timesteps=False)
	mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
	print(f"Mean reward: {mean_reward} +/- {std_reward}")
	print("saving")
	model.save(f"dqn_model{model_number}/{loaded_timesteps+steps*iters}")