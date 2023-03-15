from stable_baselines3.common.env_checker import check_env
from custom_env import racingEnv
import pygame
import numpy as np

img_num = int(input("Enter the racetrack number: "))
my_image = pygame.image.load(f'image{img_num}.png')
np_img = np.load(f"image{img_num}.npy")
config = np.load(f"config{img_num}.npy", allow_pickle=True)
env = racingEnv(np_img, my_image, config, allow=['forward','right','left','backward'])
# It will check your custom environment and output additional warnings if needed
check_env(env)