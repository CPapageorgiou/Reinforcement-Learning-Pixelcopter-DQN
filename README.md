# Pixelcopter
Reinforcement learning agent for the game Pixelcopter, created using the Python deep learning API Keras. The [Pixelcopter game environment](https://pygame-learning-environment.readthedocs.io/en/latest/user/games/pixelcopter.html) used was taken from the [PyGame Learning Environment (PLE)](https://pygame-learning-environment.readthedocs.io/en/latest/user/home.html) but it was simplified by removing the additional barriers to speed-up training of the agent. To run the code make sure that you have both Keras from Tensorflow and PyGame Learning Environment installed.

<p align="left">
<img src="Images\environment.png" alt="environment" height=400 width=500/>
</p>

## Approach
The alogirithm that was used to train the agent is deep Q-Learning with experience replay and fixed target network. Experience replay is the memorization of past experiences for reuse in the training of the neural network. At each step, the experience of the agent is stored in a replay buffer from which a set amount of experiences is sampled to be used to update the neural network. The fixed target network is a copy of the main neural network which is stored separately. This target network is used to generate the temporal difference (TD) values for a certain number of time steps. After this amount of time steps, the fixed target network copies the main neural network and performs the TD updates again. This process is repeated throughout the agent’s training. The purpose of the fixed target network is to prevent the TD target value from shifting, which, in turn, stabilizes the training process.

<img src="Images\pseudocode_deep_Q_Exp_rep_fixed_target_net.png" alt="pseudocode for deep Q Learnging with experience replay and fixed target network]"/>

### Model and Training
The model used is a sequential neural network with six hidden layers. The first hidden layer has 20 neurons, the next four hidden layers have 15 neurons each and the last one has 10 neurons. The output layer has 2 neurons and the input layer takes five input units.

<ul>
<li> Optimisation Function: Adam. </li>
<li> Loss function: Huber loss. </li>
<li> Activation function: ReLU for the hidden layers. Linear for the output layer. </li>
<li> Reward function: Every time the Pixelcopter passes a terrain block or barrier it gets a reward of +1 and each time it reaches a terminal state, it receives a negative reward of -5.
</ul>
