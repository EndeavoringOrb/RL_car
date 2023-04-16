import gym
from gym import spaces
import numpy as np
import math
import pygame
import threading

pygame.init()

class racingEnv(gym.Env):
    def __init__(self, np_img, bg_img, config, wall_reward, idle_reward, gate_reward, vel_reward, closeness_reward):
        super(racingEnv, self).__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        # Define the bounds for each dimension of the observation space
        #low = np.array([0.0] * 8 + [-float('inf'), -float('inf'), -1.0])
        #high = np.array([1000.0] * 8 + [float('inf'), float('inf'), 1.0])
        #self.action_space = spaces.Box(low=0, high=1, shape=(4,2), dtype=np.float32)
        self.action_space = spaces.Discrete(3)
        # Example for using image as input (channel-first; channel-last also works):
        # Create the Box observation space
        self.observation_space = spaces.Box(low=-1.0, high=1.0, dtype=np.float64, shape=(15,))
        self.reward = 0
        self.velocity = [0,0]
        self.friction_constant = 0.987
        self.TURN_SPEED = 5
        self.CAR_SPEED = 0.05
        self.BACK_SPEED = self.CAR_SPEED * 0.5
        self.WINDOW_WIDTH = np_img.shape[1]
        self.WINDOW_HEIGHT = np_img.shape[0]
        self.WINDOW_SIZE = [self.WINDOW_WIDTH,self.WINDOW_HEIGHT]
        self.np_img = np_img
        #self.game_window = pygame.display.set_mode(np_img.shape)
        self.bg_img = bg_img
        self._config = config
        self.config = config
        center_x = int(config[0][0])
        center_y = int(config[0][1])
        self.angle = 180
        self.car_shape = [6,15]
        self.done = False
        self.gates_original = config[1:]
        self.gates = self.gates_original
        self.vel = 0
        self.gate_remove_list = []
        self.gate_centers = self.get_gate_centers(self.gates_original)

        self.zero_count = 0
        self.step_num = 0
        self.step_num2 = 0

        self.wall_reward = wall_reward
        self.idle_reward = idle_reward
        self.gate_reward = gate_reward
        self.velocity_reward = vel_reward
        self.closeness_reward = closeness_reward

        self.old_distance = 1500

        #self.my_image = bg_img
        #og_size = self.my_image.get_size()
        #self.game_window = pygame.display.set_mode(og_size)
        #self.SCORE_FONT = pygame.font.SysFont('Arial', 30)
        #self.clock = pygame.time.Clock()
    
    def get_gate_centers(self, gates):
        centers = []
        for gate in gates:
            centers.append((sum([point[0] for point in gate]), sum([point[1] for point in gate])))
        return centers

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

    def are_lines_intersecting(self, line1, line2):
        """
        Check if two lines are intersecting, given their endpoints.
        
        Parameters:
        line1 (tuple): Endpoints of the first line in the format (x1, y1, x2, y2).
        line2 (tuple): Endpoints of the second line in the format (x1, y1, x2, y2).
        
        Returns:
        bool: True if the two lines are intersecting, False otherwise.
        """
        x1, y1, x2, y2 = line1
        x3, y3, x4, y4 = line2
        
        # Calculate the slopes and y-intercepts of the two lines
        slope1 = (y2 - y1) / (x2 - x1) if x2 - x1 != 0 else float('inf')
        slope2 = (y4 - y3) / (x4 - x3) if x4 - x3 != 0 else float('inf')
        
        yint1 = y1 - slope1 * x1 if x2 - x1 != 0 else x1
        yint2 = y3 - slope2 * x3 if x4 - x3 != 0 else x3
        
        # If the slopes are equal, the lines are either parallel or the same line
        if slope1 == slope2:
            return False
        
        # Calculate the x-coordinate of the intersection point
        if slope1 == float('inf'):
            x_int = x1
            y_int = yint2 + slope2 * (x_int - x3)
        elif slope2 == float('inf'):
            x_int = x3
            y_int = yint1 + slope1 * (x_int - x1)
        else:
            x_int = (yint2 - yint1) / (slope1 - slope2)
            y_int = yint1 + slope1 * (x_int - x1)
        
        # Check if the intersection point is within the range of the two lines
        if (x1 <= x_int <= x2 or x2 <= x_int <= x1) and (x3 <= x_int <= x4 or x4 <= x_int <= x3):
            if slope1 == float('inf') or slope2 == float('inf'):
                if (y1 <= yint2 <= y2 or y2 <= yint2 <= y1) or (y3 <= yint1 <= y4 or y4 <= yint1 <= y3):
                    return True
                else:
                    return False
            else:
                return True
        else:
            return False

    def step(self, action):
        self.reward = 0
        self.velocity[0],self.velocity[1] = self.velocity[0]*self.friction_constant,self.velocity[1]*self.friction_constant

        # Move the car
        if action == 0:  # forward
            self.velocity[0] += self.CAR_SPEED * math.sin((self.angle/180)*math.pi)
            self.velocity[1] += self.CAR_SPEED * math.cos((self.angle/180)*math.pi)
        elif action == 1:  # left
            self.angle += self.TURN_SPEED
            self.car_points = self.rotate_points(self.car_points, -self.TURN_SPEED)
        elif action == 2:  # right
            self.angle -= self.TURN_SPEED
            self.car_points = self.rotate_points(self.car_points, self.TURN_SPEED)
        elif action == 3:  # back
            self.velocity[0] -= self.BACK_SPEED * math.sin((self.angle/180)*math.pi)
            self.velocity[1] -= self.BACK_SPEED * math.cos((self.angle/180)*math.pi)
        else:
            self.reward -= 1

        self.car_points = self.translate_points(self.car_points, *self.velocity)

        # Keep car within the boundaries
        for i in range(2):
            self.velocity[i] = max(self.velocity[i], 0)
            self.velocity[i] = min(self.velocity[i], self.WINDOW_SIZE[i])
            while np.array([j < 0 or j >= self.WINDOW_SIZE[i] for j in self.car_points[:, i]]).any():
                self.velocity[i] = 0
                self.car_points[:, i] = np.clip(self.car_points[:, i], 0, self.WINDOW_SIZE[i] - 1)

        # Check collision with walls
        if not self.np_img[np.array(self.car_points, dtype=int).T].all():
            self.reward = self.wall_reward
            self.done = True

        # Check if gate is crossed
        for i, gate in enumerate(self.gates):
            if i not in self.gate_remove_list:
                car_line1 = (self.car_points[0][0], self.car_points[0][1], self.car_points[1][0], self.car_points[1][1])
                car_line2 = (self.car_points[1][0], self.car_points[1][1], self.car_points[2][0], self.car_points[2][1])
                car_line3 = (self.car_points[2][0], self.car_points[2][1], self.car_points[3][0], self.car_points[3][1])
                car_line4 = (self.car_points[3][0], self.car_points[3][1], self.car_points[0][0], self.car_points[0][1])
                gate_line = (*gate[0], *gate[-1])
                if self.are_lines_intersecting(car_line1, gate_line) or \
                self.are_lines_intersecting(car_line2, gate_line) or \
                self.are_lines_intersecting(car_line3, gate_line) or \
                self.are_lines_intersecting(car_line4, gate_line):
                    self.gate_remove_list.append(i)
                    self.reward += self.gate_reward

        if len(self.gate_remove_list) == len(self.gates):
            self.done = True

        # Update state
        relevant_gates = [self.gates[i] for i in range(len(self.gates)) if i not in self.gate_remove_list]
        self.gate_centers = self.get_gate_centers(relevant_gates)

        # Calculate distance to closest gate
        center = self.car_points.mean(axis=1).astype(int)
        center_distances = np.linalg.norm(self.gate_centers - center.reshape(1, 4), axis=1)
        closest_gate = np.argmin(center_distances)
        self.state = (self.velocity, self.angle, center_distances, closest_gate)

        return self.state, self.reward, self.done, {}


    def on_keypress(self, event):
        self.stop_flag = True

    def reset(self):
        self.step_num = 0
        self.done = False
        self.gate_remove_list = []
        self.config = self._config
        self.angle, self.velocity, self.reward = 180, [0,0], 0
        self.gates, self.car_points = self.gates_original, [(self.config[0][0]-self.car_shape[0]/2,self.config[0][1]-self.car_shape[1]/2),(self.config[0][0]+self.car_shape[0]/2,self.config[0][1]-self.car_shape[1]/2),(self.config[0][0]-self.car_shape[0]/2,self.config[0][1]+self.car_shape[1]/2),(self.config[0][0]+self.car_shape[0]/2,self.config[0][1]+self.car_shape[1]/2)]
        distances, points = self.find_distances(*[p for point in self.car_points for p in point], self.angle, self.np_img)
        center_x, center_y = sum(p[0] for p in self.car_points) / 4, sum(p[1] for p in self.car_points) / 4
        center_distances = [[i, ((center_x-gate_center[0])**2+(center_y-gate_center[1])**2)**0.5] for i, gate_center in enumerate(self.gate_centers)]
        closest_gate = min(center_distances, key=lambda x: x[1])
        center_x, center_y = [(p/500-1) for p in [center_x, center_y]]
        closest_gate = [(p/500-1) for p in self.gate_centers[closest_gate[0]]]
        normalize_velocity = lambda x: (x if abs(x) <= 4 else 4*(x/abs(x)))/2 - 1
        observation = np.array([*(distance/1414.2135623731 for distance in distances), *(normalize_velocity(v) for v in self.velocity), math.sin(self.deg_to_rad(self.angle)), center_x, center_y, *closest_gate])
        return observation
