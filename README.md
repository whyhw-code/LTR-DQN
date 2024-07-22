# LTR-DQN
Combining Sorting Learning and Reinforcement Learning for Stock Trading Strategies Based on Sell-Side Analyst Reports

# Module
## LTR
The LTR consists of four elements, which are the code for training and testing lambdarank and lambdamart on the main board data and GEM data respectively, and divides the test set based on the date, as indicated by batch. The results are obtained as ranked scores for each research paper, which are stored in the temp folder.
## DQN
DQN consists of four elements, dl_dqn.py is the Environment, agent and other classes needed to implement DQN, DQN_train is the code that implements the DQN training, and calculates the single day return by taking the sort score topk of the training set in the temp file based on the action k. DQN_test_ChiNext.py is related to the DQN_test_main.py are the test codes for GEM and Main Board data respectively, which use the corresponding trained DQN models to backtest on the test set.
