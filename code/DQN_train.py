import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from dl_dqn2 import Environment, Agent
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Train DQN model for LTR-DQN experiments")

    parser.add_argument("--bankuaicode", type=str, required=True,
                        choices=["3068", "0060"],
                        help="Market code. Example: 3068 (ChiNext), 0060 (Main Board)")

    parser.add_argument("--train_year", type=int, required=True,
                        choices=[2, 3, 4],
                        help="Number of training years")

    parser.add_argument("--dec", type=float, default=0.00015,
                        help="Epsilon decay rate")

    parser.add_argument("--lr", type=float, default=0.002,
                        help="Learning rate")

    parser.add_argument("--n_games", type=int, default=31,
                        help="Number of training episodes")

    parser.add_argument("--gamma", type=float, default=0.9,
                        help="Discount factor")

    parser.add_argument("--epsilon", type=float, default=1.0,
                        help="Initial epsilon")

    parser.add_argument("--eps_end", type=float, default=0.03,
                        help="Minimum epsilon")

    parser.add_argument("--batch_size", type=int, default=32,
                        help="Replay batch size")

    parser.add_argument("--fc1_dims", type=int, default=256,
                        help="First hidden layer size")

    parser.add_argument("--fc2_dims", type=int, default=128,
                        help="Second hidden layer size")


    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    LTR = 'ndcg'
    test_batch = 123
    lr = args.lr

    bankuaicode = args.bankuaicode
    train_year = args.train_year
    dec = args.dec
    n_games = args.n_games
    #12batch1,05batch2,09batch3
    data = pd.read_csv(f'data/dapan/{bankuaicode}merge.csv',
                       usecols=['trade_date', 'qid_date', 'open', 'high', 'low', 'close', 'vol', 'amount', 'pct_chg', 'group_len'])
    if bankuaicode == "3068":
        temp_data = pd.read_csv(f'temp/oc/batch{test_batch}/{bankuaicode}temp_train_{LTR}_train{train_year}_0.0003_0.001_0.1_6_1000.csv',  #0.1_6创业版   0.001_5主板
                            usecols=['qid_date', 'stock_code', 'real_return', 'prediction', 'close', 'pclose'])
    elif bankuaicode == "0060":
        temp_data = pd.read_csv(
            f'temp/oc/batch{test_batch}/{bankuaicode}temp_train_{LTR}_train{train_year}_0.0003_0.001_0.001_5_1000.csv',
            # 0.1_6创业版   0.001_5主板
            usecols=['qid_date', 'stock_code', 'real_return', 'prediction', 'close', 'pclose'])

    if train_year == 4:
        env = Environment(data, temp_data, start_date=20171206, end_date=20211206)
    elif train_year == 3:
        env = Environment(data, temp_data, start_date=20181206, end_date=20211206)
    elif train_year == 2:
        env = Environment(data, temp_data, start_date=20191206, end_date=20211206)

    # gamma的折扣率它必须介于0和1之间。越大，折扣越小。这意味着学习，agent 更关心长期奖励。
    # 另一方面，gamma越小，折扣越大。这意味着我们的 agent 更关心短期奖励（最近的奶酪）。
    # epsilon探索率ϵ。即策略是以1−ϵ的概率选择当前最大价值的动作，以ϵ的概率随机选择新动作。
    agent = Agent(
        gamma=args.gamma,
        epsilon=args.epsilon,
        batch_size=args.batch_size,
        n_actions=5,
        eps_end=args.eps_end,
        eps_dec=args.dec,
        input_dims=[13],
        fc1_dims=args.fc1_dims,
        fc2_dims=args.fc2_dims,
        lr = args.lr
    )
    eps_history, losss= [], []
    n_games = args.n_games # 训练局数

    for i in range(n_games):
        day_profit = 0
        day_profits, losss, funds = [], [], []
        done = False
        observation = env.reset()
        # observation = [float(val.replace('-', '')) if isinstance(val, str) and val.replace('-', '').isdigit() else val for val in observation]
        while not done:
            print("qid_date:", env.qid_date)
            action = agent.choose_action(observation)
            observation_, reward, done, real_action, select_positive_count = env.step(action)
            day_profit = env.day_profit
            fund = env.fund
            agent.store_transition(observation, action, reward, observation_, done)
            loss = agent.learn()
            observation = observation_

            # 保存一下每局的收益，最后画个图
            losss.append(loss)
            day_profits.append(day_profit)
            funds.append(fund)
            eps_history.append(agent.epsilon)
            avg_profits = np.mean(day_profits[-100:])

            print('episode', i,
                  'epsilon %.2f' % agent.epsilon,
                  'loss %.4f' % loss,
                  'day_profits %.4f' % day_profit,
                  'fund %.4f' % fund,
                  'action %s' % action,
                  'real_action %s' % real_action, '\n')

            # 保持 x 和 profits 的长度相同
            x = [j for j in range(1, len(funds) + 1)]
            # print(x, '\n')
        #
        # if i == n_games-1:
        #     plt.plot(x, funds)
        #     plt.xlabel('X')
        #     plt.ylabel('funds')
        #     plt.title('funds Over Time')
        #     plt.show()
        #     plt.plot(x, losss)
        #     plt.xlabel('X')
        #     plt.ylabel('loss')
        #     plt.title('loss Over Time')
        #     plt.savefig(f'loss_train/batch{test_batch}/{bankuaicode}_{LTR}_{train_year}year_top4_train{train_year}TEST.png')
        #     plt.show()
    agent.save_model(f'model/batch{test_batch}/{bankuaicode}_{LTR}_{train_year}year_top4_train{train_year}TESToc_{lr}')

