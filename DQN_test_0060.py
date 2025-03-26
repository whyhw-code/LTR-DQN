import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from dl_dqn2 import Environment,DeepQNetwork ,Agent

if __name__ == '__main__':
    bankuaicode = '0060'
    LTR = 'ndcg'
    train_year = 3
    test_batch = 123
    lr = 0.002
    test_start = 20211207
    data = pd.read_csv(f'data/dapan/{bankuaicode}merge.csv',
                       usecols=['trade_date', 'qid_date', 'open', 'high', 'low', 'close', 'vol', 'amount', 'pct_chg', 'group_len'])
    temp_data = pd.read_csv(f'data/temp/oc/batch{test_batch}/{bankuaicode}temp_test_{LTR}_train{train_year}_0.0003_0.001_0.001_5_1000.csv',
                            usecols=['qid_date', 'stock_code', 'real_return', 'prediction', 'close', 'pclose'])
    env = Environment(data, temp_data, start_date=20211022, end_date=20230303)
    # gamma的折扣率它必须介于0和1之间。越大，折扣越小。这意味着学习，agent 更关心长期奖励。
    # 另一方面，gamma越小，折扣越大。这意味着我们的 agent 更关心短期奖励（最近的奶酪）。
    # epsilon探索率ϵ。即策略是以1−ϵ的概率选择当前最大价值的动作，以ϵ的概率随机选择新动作。
    agent = Agent(gamma=0.9, epsilon=0.03, batch_size=32, n_actions=5, eps_end=0.03, eps_dec=0.0001, input_dims=[13], lr=lr, fc1_dims=256,fc2_dims=128)
    agent.load_model(f'model/batch{test_batch}/{bankuaicode}_{LTR}_{train_year}year_top4_train{train_year}TESToc')
    funds, day_return, real_actions, eps_history, select_positive_counts, losss, date = [], [], [], [], [], [], []
    day_profit = 0
    done = False
    observation = env.reset()

    while not done:
        print("qid_date:", env.qid_date)
        date.append(env.qid_date)
        action = agent.choose_action(observation)
        observation_, reward, done, real_action, positive_count = env.step(action)
        day_profit = env.day_profit
        fund = env.fund
        agent.store_transition(observation, action, reward, observation_, done)
        loss = agent.learn()
        observation = observation_

        # 保存一下每局的收益，最后画个图
        losss.append(loss)
        funds.append(fund)
        day_return.append(env.day_real_return)
        real_actions.append(real_action)
        eps_history.append(agent.epsilon)
        avg_profits = np.mean(funds[-100:])
        select_positive_counts.append(positive_count)

        print('epsilon %.2f' % agent.epsilon,
              'day_profit %.4f' % day_profit,
              'avg profits %.4f' % avg_profits,
              'day_return %.4f' % env.day_real_return,
              'action %s' % action,
              'real_action %s' % real_action, '\n')

        # 保持 x 和 profits 的长度相同
        x = [j for j in range(1, len(funds) + 1)]
        # print(x, '\n')

    result_data = {
        'qid_date': date,
        'funds': funds,
        'day_return': day_return,
        'real_action': real_actions,
        'select_positive_count': select_positive_counts
    }
    result_df = pd.DataFrame(result_data)
    result_df.to_excel(f'result/batch{test_batch}/{bankuaicode}_{LTR}_{train_year}year_train{train_year}_top4TESToc_{lr}xinxin.xlsx', index=False)

    results_df = result_df[test_start <= result_df['qid_date']]

    # 初始金额
    initial_amount = results_df.iloc[0]['funds']

    # 年化收益率 (ARR)
    trading_days = results_df.shape[0]
    ARR = (results_df.iloc[-1]['funds'] / initial_amount) ** (242 / trading_days) - 1

    # 最大回撤率
    results_df['cummax'] = results_df['funds'].cummax()
    results_df['drawdown'] = (results_df['funds'] - results_df['cummax']) / results_df['cummax']
    max_drawdown = results_df['drawdown'].min()

    # 卡尔玛比率
    calmar_ratio = ARR / abs(max_drawdown) if max_drawdown != 0 else np.nan

    # 夏普比率 (假设无风险利率为0.025)
    std_daily_return = results_df['day_return'].std()
    sharpe_ratio = (ARR - 0.025) / std_daily_return if std_daily_return != 0 else np.nan

    shenglv = results_df['select_positive_count'].sum() / results_df['real_action'].sum()

    # 输出结果
    print(f"年化收益率 (ARR): {ARR:.3f}")
    print(f"最大回撤率: {-max_drawdown:.3f}")
    print(f"卡尔玛比率: {calmar_ratio:.3f}")
    print(f"夏普比率: {sharpe_ratio:.3f}")
    print(f"WR: {shenglv:.3f}")

    plt.plot(x, funds)
    plt.xlabel('x')
    plt.ylabel('Funds')
    plt.title('Funds Over Time')
    plt.show()

    plt.plot(x, losss)
    plt.xlabel('x')
    plt.ylabel('loss')
    plt.title('Loss Over Time')
    plt.show()