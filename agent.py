import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Huber
from ple.games.pixelcopter import Pixelcopter
from ple import PLE
import numpy as np
import pygame
import datetime
from collections import deque
import random
import os
import matplotlib.pyplot as plt


np.random.seed(123)
random.seed(123)
tf.random.set_seed(1234)

# game environment
game = Pixelcopter(width=600, height=600)
env = PLE(game) 

# parameters
epochs = 1000
epsilon = 1
discount = 0.999
eq_weights = 20
experience_replay = deque(maxlen = 5000)
batch_size = 30

rewards = []
time_steps_per_episode = []

def create_network():
    model = Sequential()
    model.add(Dense(20, input_dim=5, activation="relu"))
    model.add(Dense(15, activation="relu"))
    model.add(Dense(15, activation="relu"))
    model.add(Dense(15, activation="relu"))
    model.add(Dense(15, activation="relu"))
    model.add(Dense(10, activation="relu")) 
    model.add(Dense(2, activation="linear"))
    adam = Adam(lr = 0.05)
    model.compile(loss=Huber(delta = 1.5), optimizer = adam, metrics = [keras.metrics.Accuracy()])

    return model

model = create_network()
target_model = create_network()

# method for training the agent using DQN with experience replay and fixed target network.
def trainAgent(epochs, rewards, discount, epsilon, batch_size):
    step = 0

    for epoch in range(epochs):
        print("epoch: ",epoch)
        env.reset_game()
        counter = 0

        while (not env.game_over()):
            step += 1
            counter+=1
            state = game.getGameState() # dictionary with 5 key-value  pairs
            state_arr = np.array([[state[k] for k in state]]) # stores the values of the dict in a numpy array
            q = model.predict(state_arr) # q value for each action (up or do nothing).
            
            action = choose_action(q)   
            
            # take the action and get the reward and the new state.
            action_arr = env.getActionSet()
            reward = env.act(action_arr[action])     
            new_state = game.getGameState()
            newstate_arr = np.array([[new_state[k] for k in state]])

            experience_replay.append((state_arr,action,reward,newstate_arr))

            # check if the experience replay has enough elements to sample. 
            if len(experience_replay) < batch_size:
                continue
            
            # get a random sample from the experience replay buffer.
            minibatch = random.sample (experience_replay, batch_size)
            input_data = np.empty((0,5))
            target_data = np.empty((0,2))

            # use the minibatch to train te agent.
            exp_rep(input_data, target_data, minibatch)

            # equalise the weights of the training network and the target network after a fixed amount of steps.
            if step % eq_weights == 0:
                equalise_weights()
        time_steps_per_episode.append(counter)
        
        rewards.append(env.score())
        epoch += 1 

        # epsilon decay.
        if epsilon > 0.001: 
            epsilon -= (1/epochs)

        if epoch % 100 == 0 and epoch!=0:    
            graphs(rewards, time_steps_per_episode, epoch)


# chooses action using epsilon-greedy policy.
def choose_action(q):
    if (np.random.uniform() < epsilon):
        action = np.random.randint(0,2)
    else:
        action = np.argmax(q[0])

    return action 

# trains the agent using experience replay, sampling from a minibatch.
def exp_rep(input_data, target_data, minibatch):

    for sample in minibatch:
        st, action, r, new_state = sample
        target = target_model.predict(st)

        if env.game_over():
            target[0][action] = r
        else:
            # update the q-value of the current state using prediction for the next q-value from the target network. 
            target[0][action] = r + discount * np.max(target_model.predict(new_state)) 
        
        input_data = np.append(input_data, st, axis=0)
        target_data = np.append(target_data, target, axis=0)

    # train the network using the input data and target data that have been collected through the for loop. 
    model.fit(input_data, target_data, epochs = 1, batch_size= batch_size, verbose = 2)

# sets the weights of the target network equal to those of the training network.
def equalise_weights():
    weights = model.get_weights()
    target_weights = target_model.get_weights()
    for i in range(len(target_weights)):
        target_weights[i] = weights[i]
    target_model.set_weights(target_weights)


trainAgent(epochs, rewards, discount, epsilon, batch_size)


# saves the model and creates graphs for epochs over time steps and epochs over reward. 
def graphs(rewards, time_steps_per_episode, epoch):

    f = f"epoch_{epoch}_model_" + datetime.datetime.now().strftime("%d-%m-%Y %H:%M").replace(":", "+")  + ".h5py"
    model.save(f)
    g = f"epoch_{epoch}_target_" + datetime.datetime.now().strftime("%d-%m-%Y %H:%M").replace(":", "+")  + ".h5py"
    target_model.save(g)

    plt.plot(rewards)
    plt.xlabel("Epochs")
    plt.ylabel("Score")
    plt.show(block=False) 
    plt.savefig(f"{epoch} epochs plot.png")
    plt.clf()

    plt.plot( time_steps_per_episode)
    plt.xlabel("Epochs")
    plt.ylabel(" time_steps")
    plt.show(block=False) 
    plt.savefig(f"{epoch} epochs time steps plot.png")
    plt.clf()