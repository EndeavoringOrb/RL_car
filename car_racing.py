import pygame
import math
import numpy as np
from scipy.ndimage import distance_transform_edt

# Initialize Pygame
pygame.init()

# Set physics things
friction_constant = 0.987

# Load the background image + make game window
img_num = 2 #int(input("Enter the image number: "))
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
car_shape = [15,30] #width,height

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
config = np.load(f"config{img_num}.npy")

# Load the car image
car_image = pygame.image.load('car.png').convert_alpha()
car_image = pygame.transform.scale(car_image, (car_shape[0], car_shape[1]))
car_image = pygame.transform.rotate(car_image, 90)
car_img_size = car_image.get_size()

# Load the background values
np_img = np.load(f"image{img_num}.npy")

# Set the initial car position
car_x = config[0][0]*WINDOW_WIDTH/og_size[0]
car_y = config[0][1]*WINDOW_HEIGHT/og_size[1]
car_points = [(config[0][0]-car_shape[0]/2,config[0][1]-car_shape[1]/2),(config[0][0]+car_shape[0]/2,config[0][1]-car_shape[1]/2),(config[0][0]-car_shape[0]/2,config[0][1]+car_shape[1]/2),(config[0][0]+car_shape[0]/2,config[0][1]+car_shape[1]/2)]

# Set the initial score
score = 0

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

# Set the clock
clock = pygame.time.Clock()

# Set the game loop
game_running = True
velocity = [0,0]
angle = 180
step_count = 0
while game_running:
    print(step_count,end="\r")
    step_count += 1
    velocity[0],velocity[1] = velocity[0]*friction_constant,velocity[1]*friction_constant

    # Handle events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            game_running = False

    # Move the car
    keys = pygame.key.get_pressed()
    if keys[pygame.K_LEFT]:
        angle += TURN_SPEED
        car_points = rotate_points(car_points,-TURN_SPEED)
    if keys[pygame.K_RIGHT]:
        angle -= TURN_SPEED
        car_points = rotate_points(car_points,TURN_SPEED)
    if keys[pygame.K_UP]:
        velocity[0] += CAR_SPEED * math.sin((angle/180)*math.pi)
        velocity[1] += CAR_SPEED * math.cos((angle/180)*math.pi)
        car_points
    if keys[pygame.K_DOWN]:
        velocity[0] -= CAR_SPEED * math.sin((angle/180)*math.pi)
        velocity[1] -= CAR_SPEED * math.cos((angle/180)*math.pi)

    car_x += velocity[0]
    car_y += velocity[1]
    car_points = translate_points(car_points,velocity[0],velocity[1])

    car_image2 = pygame.transform.rotate(car_image, angle)
    size = car_image2.get_size()

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

    if np_img[int(car_points[0,1]),int(car_points[0,0])] == 0 or np_img[int(car_points[1,1]),int(car_points[1,0])] == 0 or np_img[int(car_points[2,1]),int(car_points[2,0])] == 0 or np_img[int(car_points[3,1]),int(car_points[3,0])] == 0:
        #print("over")
        pass

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

    # Draw car bounding
    for point in car_points[:2]:
        pygame.draw.circle(game_window, (0,255,0), (point), 5)
    for point in car_points[2:]:
        pygame.draw.circle(game_window, (0,0,255), (point), 5)
    
    # Draw the score
    #score_text = SCORE_FONT.render(f'Score: {score//60}', True, WHITE)
    #game_window.blit(score_text, (10, 10))

    # Draw end point
    for point in config[1:]:
        pygame.draw.circle(game_window, (255,0,255), point, 5)

    # Update the score
    score += 1

    # Update the display
    pygame.display.update()

    # Set the frame rate
    clock.tick(60)

# Quit Pygame
pygame.quit()