import numpy as np
from PIL import ImageGrab, Image
import tkinter as tk
from PIL import ImageTk
import mouse
import keyboard
import math
from time import sleep

# stuff
img_num = 2
width = 1000
height = 1000
x_offset = 100
y_offset = 10
brush_size = 30
car_shape = [15,30]

save_first = False
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

while True:

    # Get the current mouse position and scale it to the image size
    x, y = mouse.get_position()
    x = int(x) - x_offset
    y = int(y) - y_offset

    # Check if the left mouse button is pressed
    if mouse.is_pressed(button='left'):
        values = get_circle_values(x,y,brush_size)
        print(values)
        print(values.shape)
        for value in values:
            img_array[value[0],value[1]] = 255
        # Draw a circle around the mouse position with a radius of 10 pixels
        #for i in range(-brush_size, brush_size+1):
        #    for j in range(-brush_size, brush_size+1):
        #        if x + i >= 0 and x + i < width and y + j >= 0 and y + j < height:
        #            img_array[y + j, x + i] = 1

    for i in config_arr:
        values = get_circle_values(i[0],i[1],5)
    img_array2 = img_array
    for value in values:
        img_array2[value[0],value[1]] = 128
    if config_arr[0][0] != 0 and config_arr[0][1] != 0:
        car_points = [(config_arr[0][0]-car_shape[0]/2,config_arr[0][1]-car_shape[1]/2),(config_arr[0][0]+car_shape[0]/2,config_arr[0][1]-car_shape[1]/2),(config_arr[0][0]-car_shape[0]/2,config_arr[0][1]+car_shape[1]/2),(config_arr[0][0]+car_shape[0]/2,config_arr[0][1]+car_shape[1]/2)]
        for point in car_points:
            img_array2[int(point[0]),int(point[1])] = 128
    # Create a PIL image from the NumPy array
    img = Image.fromarray(img_array2, mode='L')

    # Convert the PIL image to a Tkinter PhotoImage and display it on the canvas
    photo = ImageTk.PhotoImage(image=img)
    canvas.create_image(0, 0, image=photo, anchor='nw')

    # Update the Tkinter window
    root.update()

    # Check if the "s" key is pressed
    if keyboard.is_pressed('space'):
        # Save start config to a numpy file
        mouse_pos = mouse.get_position()
        config_arr.append([mouse_pos[0]-x_offset,mouse_pos[1]-y_offset])

    if keyboard.is_pressed('s'):
        if save_first == False:
            # Save start config to a numpy file
            mouse_pos = mouse.get_position()
            config_arr[0] = [mouse_pos[0]-x_offset,mouse_pos[1]-y_offset]
            save_first = True
            sleep(1)
        else:
            # Save the image as a PNG file
            img.save(f'image{img_num}.png')

            # Save the image as a NumPy file
            np.save(f'image{img_num}.npy', img_array)

            np.save(f"config{img_num}", config_arr)

            # Print a message to the console
            print("Image saved to image.png and image.npy")
            break