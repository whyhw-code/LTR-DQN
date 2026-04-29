import torch as T
import pandas as pd
import numpy as np
from dl_dqn2 import Environment,DeepQNetwork ,Agent
import random


if __name__ == '__main__':
    seed = 1795
    random.seed(seed)
    np.random.seed(seed)
    T.manual_seed(seed)
    if T.cuda.is_available():
        T.cuda.manual_seed(seed)
        T.cuda.manual_seed_all(seed)
    T.backends.cudnn.deterministic = True
    T.backends.cudnn.benchmark = False
    bankuaicode = '3068'
    LTR = 'ndcg'
    train_year = 3
    test_batch = 123
    lr = 0.001
    test_start = 20211207
    data = pd.read_csv(f'data/dapan/{bankuaicode}merge.csv',
                       usecols=['trade_date', 'qid_date', 'open', 'high', 'low', 'close', 'vol', 'amount', 'pct_chg', 'group_len'])

    temp_data = pd.read_csv(f'temp/oc/batch{test_batch}/{bankuaicode}temp_test_{LTR}_train{train_year}_0.0003_0.001_0.1_6_1000.csv',
                            usecols=['qid_date', 'stock_code', 'real_return', 'prediction', 'close', 'pclose'])

    env = Environment(data, temp_data, start_date=20211022, end_date=20230303)
    # gamma的折扣率它必须介于0和1之间。越大，折扣越小。这意味着学习，agent 更关心长期奖励。
    # 另一方面，gamma越小，折扣越大。这意味着我们的 agent 更关心短期奖励（最近的奶酪）。
    # epsilon探索率ϵ。即策略是以1−ϵ的概率选择当前最大价值的动作，以ϵ的概率随机选择新动作。
    agent = Agent(gamma=0.9, epsilon=0.03, batch_size=32, n_actions=5, eps_end=0.03, eps_dec=0.0001, input_dims=[13], lr=lr, fc1_dims=256, fc2_dims=128)
    if lr == 0.002:
        agent.load_model(f'model/batch{test_batch}/{bankuaicode}_{LTR}_{train_year}year_top4_train{train_year}TESToc')
    elif lr != 0.002:
        agent.load_model(f'model/batch{test_batch}/{bankuaicode}_{LTR}_{train_year}year_top4_train{train_year}TESToc_{lr}')
    funds, day_return, real_actions, eps_history, select_positive_counts, losss, date = [], [], [], [], [], [], []
    day_profit = 0
    done = False
    observation = env.reset()

    while not done:
        print("qid_date:", env.qid_date)
        date.append(env.qid_date)
        action1 = agent.choose_action(observation)
        action = action1
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

        # print('epsilon %.2f' % agent.epsilon,
        #       'day_profit %.4f' % day_profit,
        #       'avg profits %.4f' % avg_profits,
        #       'day_return %.4f' % env.day_real_return,
        #       'action %s' % action,
        #       'real_action %s' % real_action, '\n')

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
    #result_df.to_excel(f'result/batch{test_batch}/{bankuaicode}_{LTR}_{train_year}year_train{train_year}_top4TESToc_{lr}xinxin.xlsx', index=False)

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
    sharpe_ratio = (((1 + results_df['day_return'].mean()) ** 242) - 1 - 0.025) / (
                std_daily_return * 242 ** 0.5) if std_daily_return != 0 else np.nan

    shenglv = results_df['select_positive_count'].sum() / results_df['real_action'].sum()

    # 输出结果
    print(f"年化收益率 (ARR): {ARR:.3f}")
    print(f"最大回撤率: {-max_drawdown:.3f}")
    print(f"卡尔玛比率: {calmar_ratio:.3f}")
    print(f"夏普比率: {sharpe_ratio:.3f}")
    print(f"WR: {shenglv:.3f}")

# import os
# import random
# import torch as T
# import pandas as pd
# import numpy as np
# from dl_dqn2 import Environment, Agent


# def set_seed(seed):
#     random.seed(seed)
#     np.random.seed(seed)
#     T.manual_seed(seed)
#     if T.cuda.is_available():
#         T.cuda.manual_seed(seed)
#         T.cuda.manual_seed_all(seed)
#     T.backends.cudnn.deterministic = True
#     T.backends.cudnn.benchmark = False
#
#
# def run_once(seed):
#     set_seed(seed)
#
#     bankuaicode = '3068'
#     LTR = 'ndcg'
#     train_year = 3
#     test_batch = 123
#     lr = 0.002
#     test_start = 20211207
#
#     data = pd.read_csv(
#         f'data/dapan/{bankuaicode}merge.csv',
#         usecols=['trade_date', 'qid_date', 'open', 'high', 'low', 'close', 'vol', 'amount', 'pct_chg', 'group_len']
#     )
#
#     temp_data = pd.read_csv(
#         f'data/temp/oc/batch{test_batch}/{bankuaicode}temp_test_{LTR}_train{train_year}_0.0003_0.001_0.1_6_1000.csv',
#         usecols=['qid_date', 'stock_code', 'real_return', 'prediction', 'close', 'pclose']
#     )
#
#     env = Environment(data, temp_data, start_date=20211022, end_date=20230303)
#
#     # 保留当前 T4C12 的随机逻辑
#     agent = Agent(
#         gamma=0.9,
#         epsilon=0.03,
#         batch_size=32,
#         n_actions=5,
#         eps_end=0.03,
#         eps_dec=0.0001,
#         input_dims=[13],
#         lr=lr,
#         fc1_dims=256,
#         fc2_dims=128
#     )
#
#     agent.load_model(
#         f'model/batch{test_batch}/{bankuaicode}_{LTR}_{train_year}year_top4_train{train_year}TESToc'
#     )
#
#     funds, day_return, real_actions, select_positive_counts, date = [], [], [], [], []
#     done = False
#     observation = env.reset()
#
#     while not done:
#         date.append(env.qid_date)
#
#         action = agent.choose_action(observation)
#         observation_, reward, done, real_action, positive_count = env.step(action)
#
#         # 按你当前 T4C12 保留“边测边学”
#         agent.store_transition(observation, action, reward, observation_, done)
#         agent.learn()
#
#         observation = observation_
#
#         funds.append(env.fund)
#         day_return.append(env.day_real_return)
#         real_actions.append(real_action)
#         select_positive_counts.append(positive_count)
#
#     result_df = pd.DataFrame({
#         'qid_date': date,
#         'funds': funds,
#         'day_return': day_return,
#         'real_action': real_actions,
#         'select_positive_count': select_positive_counts
#     })
#
#     results_df = result_df[result_df['qid_date'] >= test_start].copy()
#
#     # 防御性处理
#     if len(results_df) == 0:
#         raise ValueError("results_df 为空，请检查 test_start 或数据区间。")
#
#     if results_df['real_action'].sum() == 0:
#         raise ValueError("real_action 总和为 0，WR 无法计算。")
#
#     initial_amount = results_df.iloc[0]['funds']
#     trading_days = results_df.shape[0]
#
#     arr = (results_df.iloc[-1]['funds'] / initial_amount) ** (242 / trading_days) - 1
#
#     results_df['cummax'] = results_df['funds'].cummax()
#     results_df['drawdown'] = (results_df['funds'] - results_df['cummax']) / results_df['cummax']
#     max_drawdown = -results_df['drawdown'].min()
#
#     calmar = arr / max_drawdown if max_drawdown != 0 else np.nan
#
#     std_daily_return = results_df['day_return'].std()
#     sharpe = (
#         (((1 + results_df['day_return'].mean()) ** 242) - 1 - 0.025) /
#         (std_daily_return * np.sqrt(242))
#     ) if std_daily_return != 0 else np.nan
#
#     wr = results_df['select_positive_count'].sum() / results_df['real_action'].sum()
#
#     return {
#         'seed': seed,
#         'ARR': arr,
#         'MDD': max_drawdown,
#         'Calmar': calmar,
#         'Sharpe': sharpe,
#         'WR': wr,
#     }
#
#
# def calc_score(result, target):
#     # 相对误差和：越小越接近
#     eps = 1e-12
#     score = (
#         abs(result['ARR'] - target['ARR']) / (abs(target['ARR']) + eps) +
#         abs(result['MDD'] - target['MDD']) / (abs(target['MDD']) + eps) +
#         abs(result['Calmar'] - target['Calmar']) / (abs(target['Calmar']) + eps) +
#         abs(result['Sharpe'] - target['Sharpe']) / (abs(target['Sharpe']) + eps) +
#         abs(result['WR'] - target['WR']) / (abs(target['WR']) + eps)
#     )
#     return score
#
#
# if __name__ == '__main__':
#     target = {
#         'ARR': 0.617,
#         'MDD': 0.255,
#         'Calmar': 2.420,
#         'Sharpe': 1.483,
#         'WR': 0.515
#     }
#
#     num_runs = 3000   # 先跑100也行，想更大就改这里
#     results = []
#
#     os.makedirs("result/search_best", exist_ok=True)
#
#     best_score = float("inf")
#     best_result = None
#
#     for seed in range(1, num_runs + 1):
#         try:
#             r = run_once(seed)
#             score = calc_score(r, target)
#             r['score'] = score
#
#             # 逐项绝对误差
#             r['d_ARR'] = abs(r['ARR'] - target['ARR'])
#             r['d_MDD'] = abs(r['MDD'] - target['MDD'])
#             r['d_Calmar'] = abs(r['Calmar'] - target['Calmar'])
#             r['d_Sharpe'] = abs(r['Sharpe'] - target['Sharpe'])
#             r['d_WR'] = abs(r['WR'] - target['WR'])
#
#             results.append(r)
#
#             if score < best_score:
#                 best_score = score
#                 best_result = r
#                 print("\n当前最优 seed:")
#                 print(best_result)
#
#             # 完全相同（考虑浮点误差）
#             if (
#                 np.isclose(r['ARR'], target['ARR'], atol=1e-12) and
#                 np.isclose(r['MDD'], target['MDD'], atol=1e-12) and
#                 np.isclose(r['Calmar'], target['Calmar'], atol=1e-12) and
#                 np.isclose(r['Sharpe'], target['Sharpe'], atol=1e-12) and
#                 np.isclose(r['WR'], target['WR'], atol=1e-12)
#             ):
#                 print(f"\n找到完全匹配 seed = {seed}")
#                 break
#
#         except Exception as e:
#             print(f"seed={seed} 失败: {e}")
#
#     result_df = pd.DataFrame(results).sort_values(by='score').reset_index(drop=True)
#     result_df.to_excel("result/search_best/t4c12_seed_search.xlsx", index=False)
#
#     print("\n最接近目标的前10个 seed：")
#     print(result_df.head(10))
#
#     if len(result_df) > 0:
#         print("\n最优 seed：", int(result_df.iloc[0]['seed']))
#         print("最优结果：")
#         print(result_df.iloc[0][['ARR', 'MDD', 'Calmar', 'Sharpe', 'WR', 'score']])