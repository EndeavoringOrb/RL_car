import numpy as np
from PIL import ImageGrab, Image
import tkinter as tk
from PIL import ImageTk
import mouse
import keyboard
import math
from time import sleep
from datetime import datetime as dt

# variables
img_num = 4
width = 1000 #width of racetrack (in pixels)
height = 1000 #height of racetrack (in pixels)
x_offset = 100
y_offset = 10
brush_size = 30 #how big your drawing circle is
car_shape = [15,30]

gates = []

save_first = True
# Create an array of zeros with the specified dimensions
img_array = np.zeros((height, width), dtype=np.uint8)

# Create a Tkinter window and canvas to display the image
root = tk.Tk()
root.geometry(f"{width}x{height}+{x_offset}+{y_offset}")
root.overrideredirect(True)
canvas = tk.Canvas(root, width=width, height=height)
canvas.pack()

config_arr = [[0,0]]

def get_circle_values(x_pos, y_pos, radius):
    """
    Returns a list of integer values inside a circle
    with the given x position, y position, and radius.
    """
    integer_values = []
    for i in range(x_pos - radius, x_pos + radius + 1):
        for j in range(y_pos - radius, y_pos + radius + 1):
            if math.sqrt((i - x_pos)**2 + (j - y_pos)**2) <= radius:
                integer_values.append([j, i])
    return np.array(integer_values)

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

key_func = lambda x: math.sqrt((x[-1][0] - x[0][0])**2 + (x[-1][1] - x[0][1])**2)

def get_single_reward_gate(dot,background_array):
    radius = int(brush_size/2)
    expand_flag = True
    distances = [] # [(x,y),distance] x,y of wall point
    while expand_flag == True:
        for i in range(dot[0] - radius, dot[0] + radius + 1):
            for j in range(dot[1] - radius, dot[1] + radius + 1):
                if background_array[j][i] == 0:
                    expand_flag = False
                    distances.append([(i,j),math.sqrt((i - dot[0])**2 + (j - dot[1])**2)])
        radius += 1
    
    lines = []

    for distance in distances:
        x1,y1,x2,y2 = distance[0][0],distance[0][1],dot[0],dot[1]

        try:
            m = (y2-y1)/(x2-x1)
        except ZeroDivisionError:
            m = 0.01
        b = -m*x1+y1
        line_func = lambda x: int(m*x+b)
        if x2 > x1:
            delta = 1
        else:
            delta = -1

        found_flag = False
        count = 0
        while found_flag == False:
            x_val = x2+count
            y_val = line_func(x_val)
            if background_array[y_val][x_val] == 0:
                found_flag = True
                opposite_val = [(x_val,y_val),math.sqrt((x_val - dot[0])**2 + (y_val - dot[1])**2)]
            else:
                count += delta

        lines.append(bresenham_line(distance[0][0],distance[0][1],opposite_val[0][0],opposite_val[0][1]))

    min_line = min(lines,key=key_func)

    return min_line

def get_reward_gates(dots,background_array):
    gates = []
    for i in dots:
        gates.append(get_single_reward_gate(i,background_array))
    return gates

values1 = []
values2 = []

while True:

    # Get the current mouse position and scale it to the image size
    x, y = mouse.get_position()
    x = int(x) - x_offset
    y = int(y) - y_offset

    # Check if the left mouse button is pressed
    if mouse.is_pressed(button='left'):
        values = get_circle_values(x,y,brush_size)
        for value in values:
            img_array[value[0],value[1]] = 255
        # Draw a circle around the mouse position with a radius of 10 pixels
        #for i in range(-brush_size, brush_size+1):
        #    for j in range(-brush_size, brush_size+1):
        #        if x + i >= 0 and x + i < width and y + j >= 0 and y + j < height:
        #            img_array[y + j, x + i] = 1

    img_array2 = img_array

    if len(gates) > 0 and len(values1) < len(gates):
        values1.append(get_circle_values(gates[-1][0][0],gates[-1][0][1],5))
        values2.append(get_circle_values(gates[-1][-1][0],gates[-1][-1][1],5))
        #gates = []

        #img_array2 = np.zeros((1000,1000))

        for value in values1[-1]:
            img_array2[value[0],value[1]] = 128
        for value in values2[-1]:
            img_array2[value[0],value[1]] = 128

    #values = []
    #for i in config_arr:
    #    values.append(get_circle_values(i[0],i[1],5))
    #for value in values:
    #    img_array2[value[0],value[1]] = 128
    #if config_arr[0][0] != 0 and config_arr[0][1] != 0:
    #    car_points = [(config_arr[0][0]-car_shape[0]/2,config_arr[0][1]-car_shape[1]/2),(config_arr[0][0]+car_shape[0]/2,config_arr[0][1]-car_shape[1]/2),(config_arr[0][0]-car_shape[0]/2,config_arr[0][1]+car_shape[1]/2),(config_arr[0][0]+car_shape[0]/2,config_arr[0][1]+car_shape[1]/2)]
    #    for point in car_points:
    #        img_array2[int(point[0]),int(point[1])] = 128
    # Create a PIL image from the NumPy array
    img = Image.fromarray(img_array2, mode='L')

    # Convert the PIL image to a Tkinter PhotoImage and display it on the canvas
    photo = ImageTk.PhotoImage(image=img)
    canvas.create_image(0, 0, image=photo, anchor='nw')

    # Update the Tkinter window
    root.update()

    # Check if the "s" key is pressed
    if keyboard.is_pressed('space') and (dt.now()-space_last).total_seconds() > 1:
        # Save start config to a numpy file
        mouse_pos = mouse.get_position()
        config_arr.append([mouse_pos[0]-x_offset,mouse_pos[1]-y_offset])
        # add gates
        gates.append(get_single_reward_gate(config_arr[-1],img_array))
        space_last = dt.now()

    if keyboard.is_pressed('s'):
        if save_first == True:
            # Save start config to a numpy file
            mouse_pos = mouse.get_position()
            config_arr[0] = [mouse_pos[0]-x_offset,mouse_pos[1]-y_offset]
            save_first = False
            sleep(1)
        else:
            config_arr[1:] = get_reward_gates(config_arr[1:],img_array)
            # Save the image as a PNG file
            img = Image.fromarray(img_array, mode='L')
            img.save(f'image{img_num}.png')

            # Save the image as a NumPy file
            np.save(f'image{img_num}.npy', img_array)

            np.save(f"config{img_num}", np.array(config_arr, dtype=object), allow_pickle=True)

            # Print a message to the console
            print("Image saved to image.png and image.npy")
            break