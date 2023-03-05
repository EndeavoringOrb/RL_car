from stable_baselines3 import PPO
import os
from custom_env import racingEnv
import time
from time import sleep
import pygame
import numpy as np

"""models_dir = f"SB3models/{int(time.time())}/"

if not os.path.exists(models_dir):
	os.makedirs(models_dir)"""

print("making env...")
sleep(5)
img_num = 2 #int(input("Enter the image number: "))
my_image = pygame.image.load(f'image{img_num}.png')
np_img = np.load(f"image{img_num}.npy")
config = np.load(f"config{img_num}.npy")
env = racingEnv(np_img, my_image, config)
load_model = True if False else False
model_number = int(input("Enter the model number: "))
steps = 2048
batch_size = steps//32
if load_model == True:
	# Load the saved model from the file path
	model_path = "ppo_model6/94208.zip"  # Replace with the file path to your saved model
	model = PPO.load(model_path)
else:
	model = PPO('MlpPolicy', env, verbose=2, n_steps=steps, batch_size=batch_size)

iters = 0
while iters < 1000000000000:
	print(iters)
	iters += 1
	model.learn(total_timesteps=steps, reset_num_timesteps=False)
	print("saving")
	model.save(f"ppo_model{model_number}/{steps*iters}")