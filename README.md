# Deep Reinforcement Learning with PyTorch
This repository is built for me to track and document what I learn and build using Maxim Lapan's Book
"Deep Reinforcement Learning Hands-On".

## Cross Entropy Method

### Reasoning
- Simplicity
- Good Convergence in simple environments

### Strategy
Our cross-entropy method is policy-based. Our nonlinear function (neural network) produces a policy which determines
for every observation which action the agent should take. The output of our network will be a probability distribution
over actions. Notice, this is very similar to a classification problem in which we assign a probability to each class.
Rather, the number of classes we have is the number of actions we are able to take. Thus, our process is as follows:

1. Input a observation to the network
2. Network outputs a probability distribution over actions
3. Perform random sampling over probability distribution

Our training process is conducted as follows:

1. Play N number of episodes using the current model and environment
2. Calculate the total reward for every episode and decide on a reward boundary. Usually, we use some percentile of
all rewards, such as 50th or 70th.
3. Throw away all episodes with a reward below the boundary.
4. Train network on the remaining "elite" episodes using observations as the input
and issued actions as the desired output.
5. Repeat from step 1 until we are satisfied with the result.

You may be wondering how we calculate loss. The targets of out networks output
are determined by the actions taken in the "elite" episodes. We are essentially teaching
the network to replicate the behavior within these "elite" episodes, as we repeat this processes
our "elite" episodes grow better and better, as does our network.

### Architecture
A single layer neural network with 128 hidden units using ReLU activation.
### Results
The network was able to converge in 50 iteration with a batch size of 16.
Note, Loss has been scaled by a factor of 100 in the following image.

![cart pole](assets/cart_pole_cross_entropy.png)

