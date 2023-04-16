import tensorflow as tf
import matplotlib.pyplot as plt
from custom_env import racingEnv
import pygame
import numpy as np
import keyboard

# Set up the environment
print("making env...")
img_num = int(input("Enter the racetrack number: "))
save_num = int(input("Enter the save number of the model: "))
max_steps = int(input("Enter max env steps: "))
my_image = pygame.image.load(f'image{img_num}.png')
np_img = np.load(f"image{img_num}.npy")
config = np.load(f"config{img_num}.npy", allow_pickle=True)


env = racingEnv(np_img, my_image, config, -1, -0.001, 2, 0, 0) # -, -, +, -, +


state_size = env.observation_space.shape[0]
action_size = env.action_space.n

# Define the hyperparameters
learning_rate = 5e-4
discount_factor = 0.95
num_episodes = 1000

load_model = input("Load model? [Y/n]: ").lower() == 'y'

if load_model == False:
    # Define the Q-network
    network_shape = [int(i) for i in input("Enter the shape of the neural network seperated by spaces .ie - (2 4 5): ").split()]
    inputs = tf.keras.layers.Input(shape=(state_size,))
    x = tf.keras.layers.Dense(network_shape[0], activation='relu')(inputs)

    # Add hidden layers to the network based on the user input
    for size in network_shape[1:]:
        x = tf.keras.layers.Dense(size, activation='relu')(x)

    outputs = tf.keras.layers.Dense(action_size, activation='sigmoid')(x)
    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)

    # Define the optimizer and loss function
    optimizer = tf.keras.optimizers.Adam(lr=learning_rate)
    loss_fn = tf.keras.losses.MeanSquaredError()

    model.compile(loss=loss_fn, optimizer=optimizer, metrics=['mae'])
else:
    # Load the saved model from the file path
    model_path = input("Enter relative model path: ")  # Replace with the file path to your saved model
    model = tf.keras.models.load_model(model_path)

    network_shape = []
    for layer in model.layers:
        try:
            network_shape.append(layer.output_shape[1])
        except IndexError:
            network_shape.append(layer.output_shape[0][1])
    network_shape = network_shape[1:-1]

    # Define the optimizer and loss function
    optimizer = tf.keras.optimizers.Adam(lr=learning_rate)
    loss_fn = tf.keras.losses.MeanSquaredError()

model.summary()

def nice_list_to_str(lst):
    ret_str = str(lst[0])
    for item in lst[1:]:
        ret_str += f", {item}"
    return ret_str

def show_graph(key):
    global show_flag
    show_flag = True

show_flag = False
continue_flag = True
keyboard.hook_key('`', show_graph)

# Train the Q-network using Q-learning
for episode in range(num_episodes):
    state = env.reset()
    done = False
    total_reward = 0
    steps = 0
    
    while not done and steps < max_steps:
        # Increment step counter
        steps += 1

        # Get the Q-values for the current state
        q_values = model.predict_on_batch(state.reshape(1, -1))[0]
        
        # Choose an action using an epsilon-greedy policy
        epsilon = 1.0 / ((episode / 50) + 10)
        if np.random.uniform() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_values)
            
        # Take the chosen action and observe the new state and reward
        next_state, reward, done, _ = env.step(action)
        print(steps)
        
        # add current reward to total
        total_reward += reward
        
        # Update the Q-value for the current state and action
        q_values_target = np.copy(q_values)
        q_values_target[action] = reward + discount_factor * np.max(model.predict(next_state.reshape(1, -1))[0])
        
        # Train the Q-network using the Q-learning update rule
        with tf.GradientTape() as tape:
            predictions = model(state.reshape(1, -1))
            loss = loss_fn(q_values_target.reshape(1, -1), predictions)
            grads = tape.gradient(loss, model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))
        
        state = next_state

        if show_flag:
            # clear previous points
            plt.clf()

            # plot the points
            try:
                plt.plot(plot_data[:, 0], plot_data[:, 1], 'ro', label="total reward") # total reward
            except NameError:
                continue_flag = False
            if continue_flag:
                show_steps = True if input("Show steps? [Y/n]: ").strip().lower() == 'y' else False
                if show_steps:
                    plt.plot(plot_data[:, 0], plot_data[:, 2], 'go', label="steps ") # steps
                #plt.plot(plot_data[:, 0], plot_data[:, 3], 'bo', label="average reward") # average reward

                # add axis labels and title
                plt.xlabel('x')
                plt.ylabel('y')
                plt.title('Training Stats')

                # add a legend
                plt.legend()

                # show the plot
                plt.show()
            show_flag = False
            continue_flag = True

    # Plot stuff
    try:
        plot_data = np.append(plot_data,np.array([[episode, total_reward, steps, total_reward/steps]]),axis=0)
    except Exception as e:
        plot_data = np.array([[episode, total_reward, steps, total_reward/steps]])

    # Print the episode score
    print(f"Episode {episode} | Total Reward = {total_reward} | Steps = {steps}")
    with open(f"models/custom_models/model_{save_num}info.txt","w",encoding="utf-8") as f:
        f.write(f"Episode {episode}\nTotal Reward = {total_reward}\nSteps = {steps}\nAverage Reward = {total_reward/steps:.3f}\nNetwork Shape: ({state_size}, {nice_list_to_str(network_shape)}, {action_size})")
    model.save(f"models/custom_models/model_{save_num}.h5")
    
env.close()
