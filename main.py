# All of the required imports 

import gym
import numpy as np
import pygame
import pickle

# Register the environment with Gym----------------------------------------------------------------------------------------------------------------------------------------------------------------


gym.register(id='MyPong-v0', entry_point='my_pong_package.my_pong_env:MyPongEnv') # Registers and locates my class in a different file
env = gym.make('MyPong-v0') # Instantiates a gym object of MyPongEnv
observation = env.reset() # Reset the environment to start a new episode


# Constants---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


HIDDEN_LAYERS = 150 # This is the constant value for the number of hidden layers there are to be in the policy network
DISCOUNT_FACTOR = .99 # This is the constant value that determines how much weight is given to future rewards compared to immediate
BATCH_SIZE = 200 # This is the constant value that defines the number of samples processed in each training iteration
DECAY_RATE = 0.99 # This is the constant value that determines how quickly the moving average of past squared gradients diminishes over time
LEARNING_RATE = 1e-4 # This is the constant value taht determines the rate at which the model's parameters are updated during training.
EPSILON = 1e-6  # This is the constant value that can be added to denominators to ensure a divide by zero error never occurs
MODEL_DIMENSION = 75 * 75 # This is the constant value that represents the dimensionality of a 300x300 grid
L2_REGULARIZATION_TERM = 1e-4 # This value helps prevent overfitting... increase if overfitting is obeserved


# Creation of the model ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


# The neural network model creation is going to be used for policy approximation. 
# The Weight_Matrix1 represents the weight matrix of the first hidden layer in the neural network. 
# The Weight_Matrix2 representst the weight matric of the output layer of the neural network

# IMPORTANT MUST ADJUST VALUE ACCORDINGLY BEFORE STARTING TO WANTS
RESUME = False # Do you want to continue from the last check point? False if not, True if you do

# Conditional that checks to see if you want to start a new model and go with a previous one
if RESUME:
    model = pickle.load(open('save_v10.p', 'rb')) # This loads in the file containing the model with the weight matrices 
else:
    model = {} # Creates a new dictionary of the name model
    model['Weight_Matrix1'] = np.random.randn(HIDDEN_LAYERS, MODEL_DIMENSION) / np.sqrt(MODEL_DIMENSION) #Generates random values... the values are drawn from a Gaussian distribution. The division normalizes the intial wieghts
    model['Weight_Matrix2'] = np.random.randn(HIDDEN_LAYERS) / np.sqrt(HIDDEN_LAYERS) #Generates random values... normalizes by dividing

grad_buffer = { k : np.zeros_like(v) for k,v in model.items() } # update buffers that add up gradients over a batch
rmsprop_cache = { k : np.zeros_like(v) for k,v in model.items() } # rmsprop memory


# Predefined Variables (Not Constant) -------------------------------------------------------------------------------------------------------------------------------------------------------------------------


prev_preprocessed_observation = None # Used to hold the previous value of the observation, set to None orginally so it is easy to tell when it has not yet been populated with an observation
running_reward = None # Used to keep a running reward through episodes, set to None orginally so it is easy to tell when it has not yet been populated with a value
observations, hidden_state, dlogs, rewards = [], [], [], [] # These are all empty list used to hold varying values while the episode is happening, used after the episode is completed
reward_sum = 0 # This will keep a running reward for each episode, begins at zero so you can use += to add to the total
episode_num = 1 # This is a counter variable to keep track of the number of episodes that the program has iterated through


# Functions---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


# Preprocess a raw observation from a custom Pong environment.
def preprocess_observation(observation):
    
    cropped_observation = observation[0:HIDDEN_LAYERS, :, :] # Crop the image to remove irrelevant parts
    downsampled_observation = cropped_observation[::2, ::2, 0] # Downsample the image by a factor of 2
    background_color = 0 # Identify the background 
    object_color = 255 # Identify the object color
    downsampled_observation[downsampled_observation == background_color] = 0 # Erase background pixels
    downsampled_observation[downsampled_observation != 0] = 1 # Set all remaining non-zero pixels to 1
    preprocessed_observation = downsampled_observation.astype(np.cfloat).ravel() # Flatten the frame

    return preprocessed_observation # This is a processed observation from the raw data observation that can now be acted on

# This function completed the sigmoid equation with a given value
def sigmoid(log_probability):

    denominator = 1.0 + np.exp(-log_probability) # Calculate the denominator in the equation 

    return 1.0 / denominator # This returns the inverse of the denominator

# This function calculates the hidden layer and the probability of going up
def calculate_probability_forward(preprocessed_observation):
    
    hidden_layer = np.dot(model['Weight_Matrix1'], preprocessed_observation) # Compute the hidden layer neuron activations
    hidden_layer[hidden_layer < 0] = 0 # ReLU nonlinearity: This zeroes out any value in the numpy array that is less than zero
    log_prob_up = np.dot(model['Weight_Matrix2'], hidden_layer) # Computes the log probability of going up
    probability = sigmoid(log_prob_up) # Call to the sigmoid function

    return probability, hidden_layer # This retunrs the probabilty of going up and the hidden layer

def discounted_rewards(rewards):

    discounted_rewards = [0] * len(rewards) # This creates a list the same length of the parameter rewards and prefills all of the values with 0
    running_add = 0 # This initilizes the running counter to zero at the beginning 

    for i in reversed(range(len(rewards))): # This for loop iterates the whole list but in reversed order
        if rewards[i] != 0: # Reset the sum, since this was a game boundary (pong specific!)
            running_add = 0
        running_add = running_add * DISCOUNT_FACTOR + rewards[i] # Update the running total with discounting and reward accumulation
        discounted_rewards[i] = running_add # Updates the new reward that is discounted to the list

    return discounted_rewards # This returns the changed rewards list that is updated with a discount factor

# Compute gradients for a neural network layer in reverse (backpropagation)
def backward_pass(vert_hidden_state, vert_dlogs, vert_observations, model):

    d_Weight_Matrix2 = np.dot(vert_hidden_state.T, vert_dlogs).ravel() # Compute gradients for the second layer's weights using transposed hidden states and dlogs
    d_hidden_state = np.outer(vert_dlogs, model['Weight_Matrix2']) # Calculate gradients for the hidden state using dlogs and the second layer's weights
    d_hidden_state[vert_hidden_state <= 0] = 0 #Backprop ReLU nonlinearity
    d_Weight_Matrix1 = np.dot(d_hidden_state.T, vert_observations) # Compute gradients for the first layer's weights using transposed hidden state and observations
    grade = {
        'Weight_Matrix1' : d_Weight_Matrix1,
        'Weight_Matrix2' : d_Weight_Matrix2
    } # Create a dictionary containing computed weight gradients for layers 1 and 2

    return grade # This returns a dictionary containing the weight matrices


# Initialize Pygame /// May be able to just get rid of the pygame in the environment -------------------------------------------------------------------------------------------------------------


# Makes a call to the environment function that initilizes the pygame environment... needed before doing anything with the pygame
env.init_pygame()
# Sets the clock for the pygame 
clock = pygame.time.Clock()

# The Game Loop
running = True
while running:
    # This will stop the pygame if the window is closed
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Capture the game screen
    screen_array = pygame.surfarray.array3d(env.screen)

    # Pass the captured screen_array to your preprocessing function
    preprocessed_observation = preprocess_observation(screen_array)
    # This creates a difference image between the two most recent observations
    difference_image_observation = preprocessed_observation - prev_preprocessed_observation if prev_preprocessed_observation is not None else np.zeros(MODEL_DIMENSION)
    # Set the previous observation to the current one for next iteration
    prev_preprocessed_observation = preprocessed_observation

    # Call to function to calculate probability of going up
    probabilty, hidden_layer = calculate_probability_forward(preprocessed_observation)

    # this is a basic thing.... Karpathy does something different maybe change to his way
    action = 0 if np.random.uniform() < probabilty else 1 # roll the dice!
    # action = 0 if .5 > probabilty else 1

    # Records values that are needed for backprop
    observations.append(difference_image_observation)
    hidden_state.append(hidden_layer)
    y = 0 if action == 1 else 1
    dlogs.append(y - probabilty)

    # This steps the pong environment and then updates the pygame display
    observation, reward, done, info = env.step(action)
    env.update_display()
    # This takes the reward and adds it to a running sum
    reward_sum += reward

    rewards.append(reward) #update with the previous reward

    # This conditional happens is the episode is done (someone scores a point)
    if done:
        episode_num += 1 #counter to keep track of how many episodes occured

        # vertically stack all of the list so they are formatted and read to be used... potentially look at eradicating this step by making them vertical list to begin with
        vert_observations = np.vstack(observations)
        vert_hidden_state = np.vstack(hidden_state)
        vert_dlogs = np.vstack(dlogs)
        vert_rewards = np.vstack(rewards)
        #Reset Array memory
        observations, hidden_state, dlogs, rewards = [], [], [], []

        #calculate the discounted rewards
        discounted_vert_rewards = discounted_rewards(vert_rewards)
        #standardize the rewards as descirbed in the blog
        discounted_vert_rewards -= np.mean(discounted_vert_rewards)
        discounted_vert_rewards /= (np.std(discounted_vert_rewards) + EPSILON)

        vert_dlogs *= discounted_vert_rewards #adjust the policy log probs based on the advantages of the actions taken
        grade = backward_pass(vert_hidden_state, vert_dlogs, vert_observations, model)
        for k in model:
            grad_buffer[k] = grad_buffer[k] + grade[k].astype('float64') #this accumulates the grade over the batch

        if episode_num % BATCH_SIZE == 0:
            for W1, W2 in model.items():
                gradient = grad_buffer[W1] #grabs the gradient
                rmsprop_cache[W1] = DECAY_RATE * rmsprop_cache[W1].astype('float64') + (1 - DECAY_RATE) * gradient**2 #this update calculates a moving average of the squared gradients, weighted by DECAY_RATE
                model[W1] = model[W1] + LEARNING_RATE * gradient.astype('float64') / (np.sqrt(rmsprop_cache[W1].astype('float64')) + EPSILON) #Updates the models parameters using the RMSprop update rule
                grad_buffer[W1] = np.zeros_like(W2) # reset to the gradient buffer

        #monitoring of the training process
        running_reward = reward_sum if running_reward is None else running_reward * .99 + reward_sum * .01
        print ('RESETTING ENVIRONMENT: Episode reward total was %f. Running Mean: %f' % (reward_sum, running_reward))
        if episode_num % 100 == 0: pickle.dump(model, open('save_v10.p', 'wb'))
        reward_sum = 0
        prev_preprocessed_observation = None

    if reward != 0: # Prints out the result from the end of an episode with the reward... if there is one
        print ('ep %d: game finished, reward: %f' % (episode_num, reward))


    clock.tick(60)  # Limit frame rate to 60 FPS

pygame.quit()