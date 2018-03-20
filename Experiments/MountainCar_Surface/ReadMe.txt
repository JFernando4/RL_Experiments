
# Results Directory Naming Convention
    FunctionApproximator_FCParameters_ReinforcementLearningMethod_RLparameters

    * Function Approximators: use intuitive acronyms with upper case instead of the full name.
        - Neural Network    =>      NN
        - TileCoder         =>      TC

    * Reinforcement Learning Method: use the name of the method with only the first letter of each word capitalized
                                     and without special characters.
        - Q(sigma)          =>      QSigma
        - Sarsa             =>      Sarsa
        - Q-Learning        =>      QLearning
        - Tree-backup       =>      TreeBackup

    * Parameters: use the first letter of the parameter name as acronym. If another parameter starts with the same
                  letter, use the second letter as well. Each parameter is followed immediately by its value. Use
                  "o" to indicate "over." List them in alphabetical order.
        - alpha = 1/10                                  =>      a1o10
        - beta = 1                                      =>      b1
        - tilings = 16                                  =>      t16
        - fully connected layer with 100 neurons        =>      f100
        - convolutional layer with 100 neurons          =>      c100
        - gamma = 0.999                                 =>      g999o1000

    Examples:
        TC_a1o6t100_QSigma_b1o2e1o10g1
            (TileCoder with alpha = 1/6, 100 tilings. Q(Sigma) agent with beta = 0.5, epsilon = 0.1, gamma = 1)
        NN_a1o100000f100f10f5_QLearning_e1o10g1
            (Neural Network with alpha = 0.00001, layer 1 = fully connected with 100 neurons, layer 2 = fully connected
             with 10 neurons, layer 3 = fully connected with 5 neurons. Q-Learning agent with epsilon = 0.1, gamma = 1)
