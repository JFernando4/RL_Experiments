
# Results Directory Naming Convention

**Template**

FunctionApproximator_FCParameters_ReinforcementLearningMethod_RLparameters

- **Function Approximators:** use intuitive acronyms with upper case instead of the full name.
   - *Examples*
      - Neural Network: NN
      - TileCoder: TC
- **Reinforcement Learning Method:** use the name of the method with only the first letter of each word capitalized and without special characters.
   - *Examples*
      - Q(sigma): QSigma
      - Sarsa: Sarsa
      - Q-Learning: QLearning
      - Tree-backup: TreeBackup
- **Parameters:** use the first letter of the parameter name as acronym. If another parameter starts with the same letter, use the second letter as well. Each parameter is followed immediately by its value. Use "o" to indicate "over." List them in alphabetical order.
   - *Examples*
      - alpha = 1/10: a1o10
      - beta = 1: b1
      - number of tilings = 16: t16
      - fully connected layer with 100 neurons: f100
      - convolutional layer with 100 neurons: c100
      - gamma = 0.999: g999o1000
      - epsilon of behavior policy = 0.1: eb1o10
      - epsilon of target policy = 0.1: et1o10
      
**Examples:**
- *TC_a1o6t100_QSigma_b1o2eb1o10g1* 
   - Function Approximator: TileCoder
      - alpha = 1/6
      - number of tilings = 100
   - RL Agent: Q(Sigma):
      - beta = 1/2
      - epsilon of behavior policy = 1/10
      - gamma = 1
- *NN_a1o100000f100f10f5_QLearning_eb1o10g1*
   - Function Approximator: Neural Network
      - alpha = 0.00001
      - layer 1 = fully connected with 100 neurons
      - layer 2 = fully connected with 10 neurons
      - layer 3 = fully connected with 5 neurons
  - RL Agent: Q-Learning
      - epsilon of behavior policy = 0.1
      - gamma = 1

## Exceptions

1. If the Results' Parent Directory contains the name of the RL method, then it is omitted from the results directory's name.
2. If there is a FixedParameters.txt file in the results directory that lists all the fixed parameters for the experiment, then those parameters are omitted from the results directory's name.

**Example:**

*FixedParameters.txt file:*
```
Fixed Parameters:
   Function Approximator: TileCoder
      alpha = 0.5
   Function Approximator: Neural Network
      alpha = 0.0001
   RL Agent: QSigma (The actual name of the algorithm being used, even if the resulting algorithm is QLearning)
      beta = 1
      epsilon behavior policy = 0.1
      epsilon target policy = 0.1
      gamma = 1
      sigma = 0.5
```

* *Parent Directory's Name:* Results_QSigma_n3
   
   *Results Directory 1 Name:* TC_t8
   
   - TileCoder with 8 tilings, RL agent as in the parent directory's name, n parameter as in the parent directory's name, and other parameters as in the *FixedParameters.txt* file.
      
   *Results Directory 2 Name:* TC_t32
   
   -TileCoder with 16 tilings, RL agent as in the parent directory's name, n parameter as in the parent directory's name, and other parameters as in the *FixedParameters.txt* file.
      
   *Results Directory 3 Name:* NN_f1000f100
   
   -Neural Network with one fully connected layer with 1000 neurons and a second fully connected layer with 100 neurons, the RL agent as in the parent directory's name, n parameter as in the parent directory's name, and the rest of the parameters as in the *FixedParameters.txt* file.
