import pygame
import math
import numpy as np
from stable_baselines3 import PPO, DQN
from custom_env import racingEnv
from time import sleep
import tensorflow as tf

# Initialize Pygame
pygame.init()

# Load the saved model from the file path
model_path = "models\custom_models\model_5.h5"  # Replace with the file path to your saved model
img_num = 4
if model_path[-3:] == "zip":
    model = DQN.load(model_path)
elif model_path[-2:] == "h5":
    model = tf.keras.models.load_model(model_path)

# Set physics things
friction_constant = 0.987

# Load the background image + make game window
my_image = pygame.image.load(f'image{img_num}.png')
og_size = my_image.get_size()
game_window = pygame.display.set_mode(og_size)

WINDOW_SIZE = og_size
WINDOW_WIDTH = WINDOW_SIZE[0]
WINDOW_HEIGHT = WINDOW_SIZE[1]

# Set the colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)

# Set the car dimensions
car_shape = [6,15] #x,y

# Set the speed of the car
CAR_SPEED = 0.05
TURN_SPEED = 5

# Collision parameters
BUFFER_SIZE = 5

# Set the font for the score
SCORE_FONT = pygame.font.SysFont('Arial', 30)

# Set the window caption
pygame.display.set_caption('Car Racing Game')

# Load config
config = np.load(f"config{img_num}.npy", allow_pickle=True)

# Load the background values
np_img = np.load(f"image{img_num}.npy")

# Set the initial car position
car_x = config[0][0]*WINDOW_WIDTH/og_size[0]
car_y = config[0][1]*WINDOW_HEIGHT/og_size[1]
car_points = [(config[0][0]-car_shape[0]/2,config[0][1]-car_shape[1]/2),(config[0][0]+car_shape[0]/2,config[0][1]-car_shape[1]/2),(config[0][0]-car_shape[0]/2,config[0][1]+car_shape[1]/2),(config[0][0]+car_shape[0]/2,config[0][1]+car_shape[1]/2)]

# Set the initial score
score = 0

def get_gate_centers(gates):
    centers = []
    for gate in gates:
        centers.append((sum([point[0] for point in gate])/len(gate), sum([point[1] for point in gate])/len(gate)))
    return centers

def deg_to_rad(angle):
    return angle * math.pi / 180

def rad_to_deg(angle):
    return angle * 180 / math.pi

def find_distances(carx1, cary1, carx2, cary2, carx3, cary3, carx4, cary4, car_angle, np_img):
    center_x = int((carx1 + carx2 + carx3 + carx4) / 4)
    center_y = int((cary1 + cary2 + cary3 + cary4) / 4)
    angles = np.linspace(deg_to_rad(-car_angle), 2*math.pi+deg_to_rad(-car_angle), num=8, endpoint=False)
    distances = []
    points = []
    for angle in angles:
        endpoint_x = int(center_x + 1000 * math.cos(angle))
        endpoint_y = int(center_y + 1000 * math.sin(angle))
        line = bresenham_line(center_x, center_y, endpoint_x, endpoint_y)
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

def bresenham_line(x0, y0, x1, y1):
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

def rotate_points(points, angle_degrees):
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

def translate_points(points, move_amount_x, move_amount_y):
    translated_points = []
    for point in points:
        translated_points.append([point[0]+move_amount_x,point[1]+move_amount_y])
    return np.array(translated_points)

def are_lines_intersecting(line1, line2):
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

# Set the clock
clock = pygame.time.Clock()

# Set the game loop
game_running = True
game_over = False
velocity = [0,0]
angle = 180
gates_original = config[1:]
gates = gates_original
gate_centers = get_gate_centers(gates_original)
distances, points = find_distances(car_points[0][0],car_points[0][1],car_points[1][0],car_points[1][1],car_points[2][0],car_points[2][1],car_points[3][0],car_points[3][1],angle,np_img)
gate_remove_list = []
allow=['forward','right','left','backward']
steps = 0
while game_running:
    if game_over == True:
        steps = 0
        gate_remove_list = []
        gates = gates_original
        score = 0
        angle = 180
        velocity = [0,0]
        game_over = False
        car_x = config[0][0]*WINDOW_WIDTH/og_size[0]
        car_y = config[0][1]*WINDOW_HEIGHT/og_size[1]
        car_points = [(config[0][0]-car_shape[0]/2,config[0][1]-car_shape[1]/2),(config[0][0]+car_shape[0]/2,config[0][1]-car_shape[1]/2),(config[0][0]-car_shape[0]/2,config[0][1]+car_shape[1]/2),(config[0][0]+car_shape[0]/2,config[0][1]+car_shape[1]/2)]
    velocity[0],velocity[1] = velocity[0]*friction_constant,velocity[1]*friction_constant
    steps += 1

    relevant_gates = [gate for i, gate in enumerate(gates) if i not in gate_remove_list]
    gate_centers = get_gate_centers(relevant_gates)

    # find distance to closest gate
    center_distances = []
    center_x = int((car_points[0][0] + car_points[1][0] + car_points[2][0] + car_points[3][0]) / 4)
    center_y = int((car_points[0][1] + car_points[1][1] + car_points[2][1] + car_points[3][1]) / 4)
    for i, gate_center in enumerate(gate_centers):
        center_distances.append([i, ((center_x-gate_center[0])**2+(center_y-gate_center[1])**2)**0.5])
    closest_gate = min(center_distances, key=lambda x: x[1])

    center_x /= 1000
    center_y /= 1000
    closest_gate = [gate_centers[closest_gate[0]][0]/1000, gate_centers[closest_gate[0]][1]/1000]

    normalized_distances = [distance/1414.2135623731 for distance in distances]
    normalize_velocity = lambda x: (x if abs(x) <= 4 else 4*(x/abs(x)))/2 - 1
    normalized_velocity = [normalize_velocity(velocity[0]),normalize_velocity(velocity[1])]

    observation = np.array([*normalized_distances, normalized_velocity[0], normalized_velocity[1], math.sin(deg_to_rad(angle)), center_x, center_y, closest_gate[0], closest_gate[1]])
    # Move the car
    # action : w,a,s,d,wa,wd,sa,sd
    if model_path[-2:] == "h5":
        obs = np.array([observation])
        action = model.predict_on_batch(obs)
    else:
        action, _ = model.predict(observation)
    action_done = False
    current_sum = sum(action[0])
    print(f"{(velocity[0]**2+velocity[1]**2)**0.5:.4f} {action[0][0]/current_sum:.4f} {action[0][1]/current_sum:.4f} {action[0][2]/current_sum:.4f}",end = " ")
    action = np.argmax(action[0])
    if action == 0: #forward
        print("forward ",end="\r")
        velocity[0] += CAR_SPEED * math.sin((angle/180)*math.pi)
        velocity[1] += CAR_SPEED * math.cos((angle/180)*math.pi)
        action_done = True
        #reward += 0.1
    elif action == 1: #left
        print("left    ",end="\r")
        angle += TURN_SPEED
        car_points = rotate_points(car_points,-TURN_SPEED)
        action_done = True
    elif action == 2: #right
        print("right   ",end="\r")
        angle -= TURN_SPEED
        car_points = rotate_points(car_points,TURN_SPEED)
        action_done = True
    elif action == 3: #back
        print("backward",end="\r")
        velocity[0] -= CAR_SPEED * math.sin((angle/180)*math.pi)
        velocity[1] -= CAR_SPEED * math.cos((angle/180)*math.pi)
        action_done = True
    elif action == 4 and (allow == 'all' or 'forward left' in allow): #forward left
        velocity[0] += CAR_SPEED * math.sin((angle/180)*math.pi)
        velocity[1] += CAR_SPEED * math.cos((angle/180)*math.pi)
        angle += TURN_SPEED
        car_points = rotate_points(car_points,-TURN_SPEED)
        #reward += 0.1
        action_done = True
    elif action == 5 and (allow == 'all' or 'forward right' in allow): #forward right
        velocity[0] += CAR_SPEED * math.sin((angle/180)*math.pi)
        velocity[1] += CAR_SPEED * math.cos((angle/180)*math.pi)
        angle -= TURN_SPEED
        car_points = rotate_points(car_points,TURN_SPEED)
        #reward += 0.1
        action_done = True
    elif action == 6 and (allow == 'all' or 'backward left' in allow): #backward left
        velocity[0] -= CAR_SPEED * math.sin((angle/180)*math.pi)
        velocity[1] -= CAR_SPEED * math.cos((angle/180)*math.pi)
        angle += TURN_SPEED
        car_points = rotate_points(car_points,-TURN_SPEED)
        action_done = True
    elif action == 7 and (allow == 'all' or 'backward right' in allow): #backward right
        velocity[0] -= CAR_SPEED * math.sin((angle/180)*math.pi)
        velocity[1] -= CAR_SPEED * math.cos((angle/180)*math.pi)
        angle -= TURN_SPEED
        car_points = rotate_points(car_points,TURN_SPEED)
        action_done = True

    car_x += velocity[0]
    car_y += velocity[1]
    car_points = translate_points(car_points,velocity[0],velocity[1])

    while np.array([i < 0 for i in car_points[:,0]]).any():
        car_points = translate_points(car_points,1,0)
        velocity[0] = 0
    while np.array([i >= WINDOW_WIDTH for i in car_points[:,0]]).any():
        car_points = translate_points(car_points,-1,0)
        velocity[0] = 0
    while np.array([i < 0 for i in car_points[:,1]]).any():
        car_points = translate_points(car_points,0,1)
        velocity[1] = 0
    while np.array([i >= WINDOW_HEIGHT for i in car_points[:,1]]).any():
        car_points = translate_points(car_points,0,-1)
        velocity[1] = 0

    if len(gates) < 1:
        gates = gates_original
        gate_remove_list = []

    # 0,1 - 1,3 - 3,2 - 2,1
    line1 = (car_points[0][0],car_points[0][1],car_points[1][0],car_points[1][1])
    line2 = (car_points[1][0],car_points[1][1],car_points[3][0],car_points[3][1])
    line3 = (car_points[3][0],car_points[3][1],car_points[2][0],car_points[2][1])
    line4 = (car_points[2][0],car_points[2][1],car_points[0][0],car_points[0][1])
    for line in [line1,line2,line3,line4]:
        for i, gate in enumerate(gates):
            if are_lines_intersecting(line,(gate[0][0],gate[0][1],gate[-1][0],gate[-1][1])) and i not in gate_remove_list:
                score += 1
                gate_remove_list.append(i)
    #gates = np.delete(gates,gate_remove_list,axis=0)

    if np_img[int(car_points[0,1]),int(car_points[0,0])] == 0 or np_img[int(car_points[1,1]),int(car_points[1,0])] == 0 or np_img[int(car_points[2,1]),int(car_points[2,0])] == 0 or np_img[int(car_points[3,1]),int(car_points[3,0])] == 0:
        game_over = True

    # Clear the screen
    game_window.fill(WHITE)

    # Draw the background
    game_window.blit(my_image, (0,0))

    # Draw car png
    #game_window.blit(car_image2, (car_points[0][0], car_points[0][1]))

    # Draw the sensors
    distances, points = find_distances(car_points[0,0],car_points[0,1],car_points[1,0],car_points[1,1],car_points[2,0],car_points[2,1],car_points[3,0],car_points[3,1],angle,np_img)
    for point in points:
        pygame.draw.circle(game_window, (255,0,0), (point), 5)
    
    # Draw the score
    score_text = SCORE_FONT.render(f'Score: {score} - Steps: {steps}', True, WHITE)
    game_window.blit(score_text, (10, 10))

    # Draw lines
    for i, line in enumerate(config[1:]):
        if i in gate_remove_list:
            color = (255,0,0)
        else:
            color = (0,255,0)
        for point in line:
            pygame.draw.circle(game_window, color, point, 5)

    # Draw car bounding
    pygame.draw.circle(game_window, (0,255,0), (car_points[0][0],car_points[0][1]), 5)
    pygame.draw.circle(game_window, (255,0,0), (car_points[1][0],car_points[1][1]), 5)
    pygame.draw.circle(game_window, (0,0,0), (car_points[2][0],car_points[2][1]), 5)
    pygame.draw.circle(game_window, (0,0,255), (car_points[3][0],car_points[3][1]), 5)

    # draw closest gate
    pygame.draw.circle(game_window, (0,0,255), (closest_gate[0]*1000,closest_gate[1]*1000), 7)
    # Update the score
    #score += 1

    # Update the display
    pygame.display.update()

    # Set the frame rate
    clock.tick(24)

# Quit Pygame
pygame.quit()