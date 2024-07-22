import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from dl_dqn import Environment,DeepQNetwork ,Agent

if __name__ == '__main__':
    bankuaicode = '3068'
    LTR = 'ndcg11'
    train_year = 4
    test_batch = 2
    data = pd.read_csv(f'data/dapan/{bankuaicode}merge.csv',usecols=['trade_date', 'qid_date', 'open', 'high', 'low', 'close', 'vol', 'amount', 'pct_chg', 'group_len'])
    temp_data = pd.read_csv(f'data/temp/batch{test_batch}/{bankuaicode}temp_test_{LTR}_train{train_year}.csv',usecols=['qid_date', 'stock_code', 'real_return', 'prediction'])
    env = Environment(data, temp_data, start_date=20220318, end_date=20220929)
    # gamma的折扣率它必须介于0和1之间。越大，折扣越小。这意味着学习，agent 更关心长期奖励。
    # 另一方面，gamma越小，折扣越大。这意味着我们的 agent 更关心短期奖励（最近的奶酪）。
    # epsilon探索率ϵ。即策略是以1−ϵ的概率选择当前最大价值的动作，以ϵ的概率随机选择新动作。
    agent = Agent(gamma=0.9, epsilon=0.03, batch_size=32, n_actions=5, eps_end=0.03, eps_dec=0.0001, input_dims=[13], lr=0.002, fc1_dims=256,fc2_dims=128)
    agent.load_model(f'model/batch{test_batch}/{bankuaicode}_{LTR}_{train_year}year_top4_train{train_year}')
    profits, day_return, real_actions, eps_history, select_positive_counts, losss, date = [], [], [], [], [], [], []
    profit = 0
    done = False
    observation = env.reset()

    while not done:
        print("qid_date:", env.qid_date)
        date.append(env.qid_date)
        action = agent.choose_action(observation)
        observation_, reward, done, real_action, positive_count = env.step(action)
        profit = env.total_profit
        agent.store_transition(observation, action, reward, observation_, done)
        loss = agent.learn()
        observation = observation_

        # 保存一下每局的收益，最后画个图
        losss.append(loss)
        profits.append(profit)
        day_return.append(env.day_real_return)
        real_actions.append(real_action)
        eps_history.append(agent.epsilon)
        avg_profits = np.mean(profits[-100:])
        select_positive_counts.append(positive_count)

        print('epsilon %.2f' % agent.epsilon,
              'profits %.4f' % profit,
              'avg profits %.4f' % avg_profits,
              'day_return %.4f' % env.day_real_return,
              'action %s' % action,
              'real_action %s' % real_action, '\n')

        # 保持 x 和 profits 的长度相同
        x = [j for j in range(1, len(profits) + 1)]
        # print(x, '\n')

    result_data = {
        'qid_date': date,
        'total_profits': profits,
        'day_return': day_return,
        'real_action': real_actions,
        'select_positive_count': select_positive_counts
    }
    result_df = pd.DataFrame(result_data)
    result_df.to_excel(f'result/batch{test_batch}/{bankuaicode}_{LTR}_{train_year}year_train{train_year}_top4TEST.xlsx', index=False)

    plt.plot(x, profits)
    plt.xlabel('x')
    plt.ylabel('Profits')
    plt.title('Profits Over Time')
    plt.show()
    # plt.plot(x, losss)
    # plt.xlabel('x')
    # plt.ylabel('loss')
    # plt.title('Loss Over Time')
    # plt.savefig(f'loss_test/batch{test_batch}/{bankuaicode}_{LTR}_{train_year}year_top4_train{train_year}TEST.png')
    # plt.show()
