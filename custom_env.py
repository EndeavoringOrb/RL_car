import gym
from gym import spaces
import numpy as np
import math
import pygame

pygame.init()

class racingEnv(gym.Env):
    def __init__(self, np_img, bg_img, config):
        super(racingEnv, self).__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        low = np.concatenate((np.zeros(20), np.array([-500, -500]), np.array([0])))
        high = np.concatenate((np.ones(20), np.array([500, 500]), np.array([1])))
        self.action_space = spaces.Box(low=0, high=1, shape=(4,2), dtype=np.float32)
        # Example for using image as input (channel-first; channel-last also works):
        self.observation_space = spaces.Box(low=0, high=1000, shape=(8,), dtype=np.float32)
        self.reward = 0
        self.velocity = [0,0]
        self.friction_constant = 0.987
        self.TURN_SPEED = 5
        self.CAR_SPEED = 0.05
        self.WINDOW_WIDTH = np_img.shape[1]
        self.WINDOW_HEIGHT = np_img.shape[0]
        self.np_img = np_img
        #self.game_window = pygame.display.set_mode(np_img.shape)
        self.bg_img = bg_img
        self._config = config
        self.config = config
        center_x = int(config[0][0])
        center_y = int(config[0][1])
        self.max_distance = ((self.config[-1][0]-center_x)**2 + (self.config[-1][1]-center_y)**2)**0.5
        self.angle = 180
        self.car_shape = [6,15]
        self.done = False
    
    def deg_to_rad(self, angle):
        return angle * math.pi / 180

    def rad_to_deg(self, angle):
        return angle * 180 / math.pi

    def find_distances(self, carx1, cary1, carx2, cary2, carx3, cary3, carx4, cary4, car_angle, np_img):
        center_x = int((carx1 + carx2 + carx3 + carx4) / 4)
        center_y = int((cary1 + cary2 + cary3 + cary4) / 4)
        angles = np.linspace(self.deg_to_rad(-car_angle), 2*math.pi+self.deg_to_rad(-car_angle), num=8, endpoint=False)
        distances = []
        points = []
        for angle in angles:
            endpoint_x = int(center_x + 1000 * math.cos(angle))
            endpoint_y = int(center_y + 1000 * math.sin(angle))
            line = self.bresenham_line(center_x, center_y, endpoint_x, endpoint_y)
            for point in line:
                try:
                    if np_img[point[1], point[0]] == 0:
                        distance = ((point[0] - center_x)**2 + (point[1] - center_y)**2)**0.5
                        distances.append(distance)
                        points.append((point))
                        break
                except IndexError:
                    pass
            else:
                distances.append(-10)  # no black pixels found, set distance to max
                points.append((-10,-10))
        return distances,points

    def bresenham_line(self, x0, y0, x1, y1):
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = -1 if x0 > x1 else 1
        sy = -1 if y0 > y1 else 1
        err = dx - dy
        line = []
        while True:
            line.append((x0, y0))
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy
        return line

    def rotate_points(self, points, angle_degrees):
        # Find the center point of the four points
        center_x = sum([p[0] for p in points]) / len(points)
        center_y = sum([p[1] for p in points]) / len(points)

        # Convert the rotation angle from degrees to radians
        angle_rad = math.radians(angle_degrees)

        # Create a rotation matrix
        rotation_matrix = [
            [math.cos(angle_rad), -math.sin(angle_rad)],
            [math.sin(angle_rad), math.cos(angle_rad)]
        ]

        # Rotate each point around the center point
        rotated_points = []
        for p in points:
            # Translate the point to the origin
            x = p[0] - center_x
            y = p[1] - center_y

            # Apply the rotation matrix
            rotated_x = x * rotation_matrix[0][0] + y * rotation_matrix[0][1]
            rotated_y = x * rotation_matrix[1][0] + y * rotation_matrix[1][1]

            # Translate the point back to its original position
            rotated_x += center_x
            rotated_y += center_y

            rotated_points.append([rotated_x, rotated_y])

        return np.array(rotated_points)

    def translate_points(self, points, move_amount_x, move_amount_y):
        translated_points = []
        for point in points:
            translated_points.append([point[0]+move_amount_x,point[1]+move_amount_y])
        return np.array(translated_points)
        
    def convert_to_int(self, array):
        array = list(array) #array[0]
        largest_val = max(array)
        number = array.index(largest_val)
        if number == 10:
            return "N"
        return number

    def get_coordinates_in_box(self, x1, y1, x2, y2):
        """
        Returns an array of all the x,y coordinates contained within the box defined by the
        coordinates (x1, y1) and (x2, y2), where (x1, y1) represents one corner
        and (x2, y2) represents the opposite corner.
        """
        # Compute the slope and y-intercept of the line passing through (x1, y1) and (x2, y2)
        if x1 != x2:
            m = (y2 - y1) / (x2 - x1)
            b = y1 - m * x1
        else:
            m = float('inf')
            b = None

        # Determine the range of x and y values to iterate through based on the corners
        x_min = int(min(x1, x2))
        x_max = int(max(x1, x2))
        y_min = int(min(y1, y2))
        y_max = int(max(y1, y2))

        # Iterate through all possible x,y coordinates and check if they are inside the box
        coordinates = []
        for x in range(x_min, x_max + 1):
            if x == x1 and x == x2:
                # Skip this x value since both corners lie on the same vertical line
                continue
            if m == float('inf'):
                # Handle the case where the line is vertical
                y_range = (y_min, y_max)
            else:
                # Compute the range of y values for the current x value
                y_range = (m * x + b, m * x + b)
                if x == x1:
                    y_range = (y1, y_range[1])
                elif x == x2:
                    y_range = (y_range[0], y2)
                y_range = (int(min(y_range)), int(max(y_range)))
            for y in range(y_range[0], y_range[1] + 1):
                coordinates.append([x, y])

        return coordinates

    def arr_in_arr(self, smaller_arr, larger_arr):
        ret_list = []
        for i in range(len(larger_arr)):
            if smaller_arr[0] == larger_arr[i][0] and smaller_arr[1] == larger_arr[i][1]:
                ret_list.append(i)
        return ret_list

    def step(self, action):
        #self.reward = 0
        self.velocity[0],self.velocity[1] = self.velocity[0]*self.friction_constant,self.velocity[1]*self.friction_constant

        # Move the car
        if action[0][0] > action[0][1]:
            self.angle += self.TURN_SPEED
            self.car_points = self.rotate_points(self.car_points,-self.TURN_SPEED)
        if action[1][0] > action[1][1]:
            self.angle -= self.TURN_SPEED
            self.car_points = self.rotate_points(self.car_points,self.TURN_SPEED)
        if action[2][0] > action[2][1]:
            self.velocity[0] += self.CAR_SPEED * math.sin((self.angle/180)*math.pi)
            self.velocity[1] += self.CAR_SPEED * math.cos((self.angle/180)*math.pi)
        else:
            self.reward -= 0.05
        if action[3][0] > action[3][1]:
            self.velocity[0] -= self.CAR_SPEED * math.sin((self.angle/180)*math.pi)
            self.velocity[1] -= self.CAR_SPEED * math.cos((self.angle/180)*math.pi)
            self.reward -= 0.5

        self.car_points = self.translate_points(self.car_points,self.velocity[0],self.velocity[1])

        while np.array([i < 0 for i in self.car_points[:,0]]).any():
            self.car_points = self.translate_points(self.car_points,1,0)
            self.velocity[0] = 0
        while np.array([i >= self.WINDOW_WIDTH for i in self.car_points[:,0]]).any():
            self.car_points = self.translate_points(self.car_points,-1,0)
            self.velocity[0] = 0
        while np.array([i < 0 for i in self.car_points[:,1]]).any():
            self.car_points = self.translate_points(self.car_points,0,1)
            self.velocity[1] = 0
        while np.array([i >= self.WINDOW_HEIGHT for i in self.car_points[:,1]]).any():
            self.car_points = self.translate_points(self.car_points,0,-1)
            self.velocity[1] = 0

        remove_list = []
        coords = self.get_coordinates_in_box(self.car_points[0][0],self.car_points[0][1],self.car_points[3][0],self.car_points[3][1])
        for i in range(len(self.config[1:])):
            dot_list = self.arr_in_arr(self.config[i+1],coords)
            remove_list.extend(dot_list)
        self.config = np.delete(self.config, remove_list, 0)
        for j in range(len(remove_list)):
            self.reward += 2

        # Draw the sensors
        distances, points = self.find_distances(self.car_points[0,0],self.car_points[0,1],self.car_points[1,0],self.car_points[1,1],self.car_points[2,0],self.car_points[2,1],self.car_points[3,0],self.car_points[3,1],self.angle,self.np_img)

        observation = np.array(distances)

        center_x = int((self.car_points[0,0] + self.car_points[1,0] + self.car_points[2,0] + self.car_points[3,0]) / 4)
        center_y = int((self.car_points[0,1] + self.car_points[1,1] + self.car_points[2,1] + self.car_points[3,1]) / 4)
        end_distance = ((self.config[-1][0]-center_x)**2 + (self.config[-1][1]-center_y)**2)**0.5

        info = {}

        if self.np_img[int(self.car_points[0,1]),int(self.car_points[0,0])] == 0 or self.np_img[int(self.car_points[1,1]),int(self.car_points[1,0])] == 0 or self.np_img[int(self.car_points[2,1]),int(self.car_points[2,0])] == 0 or self.np_img[int(self.car_points[3,1]),int(self.car_points[3,0])] == 0:
            self.reward -= 1
            self.done = True
            self.reset()

        return observation, self.reward, self.done, info

    def on_keypress(self, event):
        self.stop_flag = True

    def reset(self):
        self.config = self._config
        self.angle = 180
        self.velocity = [0,0]
        self.reward = 0
        self.car_points = [(self.config[0][0]-self.car_shape[0]/2,self.config[0][1]-self.car_shape[1]/2),(self.config[0][0]+self.car_shape[0]/2,self.config[0][1]-self.car_shape[1]/2),(self.config[0][0]-self.car_shape[0]/2,self.config[0][1]+self.car_shape[1]/2),(self.config[0][0]+self.car_shape[0]/2,self.config[0][1]+self.car_shape[1]/2)]
        distances, points = self.find_distances(self.car_points[0][0],self.car_points[0][1],self.car_points[1][0],self.car_points[1][1],self.car_points[2][0],self.car_points[2][1],self.car_points[3][0],self.car_points[3][1],self.angle,self.np_img)
        observation = distances
        return observation