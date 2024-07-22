# LTR-DQN
Combining Sorting Learning and Reinforcement Learning for Stock Trading Strategies Based on Sell-Side Analyst Reports

# 各模块
LTR包括四个内容，分别是在主板数据和创业板数据上进行lambdarank和lambdamart的训练和测试的代码，并且基于日期划分了测试集，如batch表示。结果得到每篇研报的排序得分，存储在temp文件夹中。
DQN包括四个内容，dl_dqn.py是实现DQN所需要的Environment，agent等类，DQN_train是实现DQN训练的代码，根据action k取temp文件中的训练集的排序得分topk计算单日收益率。DQN_test_ChiNext.py与DQN_test_main.py分别是创业板与主板数据的测试代码，使用相对应训练好的DQN模型，在测试集上进行回测。
