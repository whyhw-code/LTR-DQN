import torch as T
import pandas as pd
import numpy as np
from dl_dqn2 import Environment, DeepQNetwork, Agent, T4ExcelWriter
import random
from pathlib import Path
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=0.002)
parser.add_argument('--train_year', type=int, default=3)
parser.add_argument('--test_batch', type=int, default=123)
parser.add_argument('--write_excel', action='store_true')
args = parser.parse_args()


def to_int_date(x):
    """
    统一日期格式为 YYYYMMDD 整数。
    兼容：
    20211207
    2021-12-07
    2021/12/7
    20211207.0
    Timestamp
    """
    if pd.isna(x):
        return None

    if hasattr(x, "strftime"):
        return int(x.strftime("%Y%m%d"))

    x = str(x).strip()

    if x.endswith(".0"):
        x = x[:-2]

    x2 = x.replace("-", "").replace("/", "")

    if x2.isdigit() and len(x2) == 8:
        return int(x2)

    return int(pd.to_datetime(x).strftime("%Y%m%d"))


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
    train_year = args.train_year
    test_batch = args.test_batch
    lr = args.lr
    test_start = 20211207

    # =====================================================
    # 关键逻辑：
    # lr == 0.002：用于 T4 / Fig5 复现，固定每日选择结果
    # lr != 0.002：用于 F3b 敏感性分析，恢复 DQN 动态 action
    # =====================================================
    use_fixed_action = (lr == 0.002)

    fixed_action_map = None

    if use_fixed_action:
        file_map = {
            3: "meiri_xuanze.csv",
            2: "meiri_xuanze2.csv",
            4: "meiri_xuanze4.csv",
        }

        CODE_DIR = Path(__file__).resolve().parent
        fixed_action_file = file_map.get(train_year, f"meiri_xuanze{train_year}.csv")
        fixed_action_path = CODE_DIR / "temp" / "oc" / f"batch{test_batch}" / fixed_action_file

        if not fixed_action_path.exists():
            raise FileNotFoundError(f"找不到每日选择文件：{fixed_action_path}")

        try:
            daily_select = pd.read_csv(fixed_action_path, encoding="utf-8-sig")
        except UnicodeDecodeError:
            daily_select = pd.read_csv(fixed_action_path, encoding="gbk")

        daily_select.columns = daily_select.columns.astype(str).str.strip()

        if "qid_date" not in daily_select.columns:
            raise ValueError(f"每日选择文件中找不到 qid_date 列，当前列名：{daily_select.columns.tolist()}")

        daily_select["_qid_date"] = daily_select["qid_date"].map(to_int_date)
        daily_select = daily_select.dropna(subset=["_qid_date"]).copy()
        daily_select["_qid_date"] = daily_select["_qid_date"].astype(int)

        if bankuaicode in daily_select.columns:
            action_col = bankuaicode
        elif bankuaicode.lstrip("0") in daily_select.columns:
            action_col = bankuaicode.lstrip("0")
        else:
            raise ValueError(
                f"每日选择文件中找不到 {bankuaicode} 对应的列，"
                f"当前列名：{daily_select.columns.tolist()}"
            )

        daily_select[action_col] = (
            pd.to_numeric(daily_select[action_col], errors="coerce")
            .fillna(0)
            .clip(0, 4)
            .astype(int)
        )

        fixed_action_map = dict(
            zip(
                daily_select["_qid_date"],
                daily_select[action_col]
            )
        )

    data = pd.read_csv(
        f'data/dapan/{bankuaicode}merge.csv',
        usecols=[
            'trade_date', 'qid_date',
            'open', 'high', 'low', 'close',
            'vol', 'amount', 'pct_chg', 'group_len'
        ]
    )

    temp_data = pd.read_csv(
        f'temp/oc/batch{test_batch}/{bankuaicode}temp_test_{LTR}_train{train_year}_0.0003_0.001_0.1_6_1000.csv',
        usecols=[
            'qid_date', 'stock_code', 'real_return',
            'prediction', 'close', 'pclose'
        ]
    )

    # lr == 0.002：固定 action，直接从测试期开始
    # lr != 0.002：敏感性分析，恢复原始代码从 20211022 开始
    if use_fixed_action:
        env_start_date = test_start
    else:
        env_start_date = 20211022

    env = Environment(data, temp_data, start_date=env_start_date, end_date=20230303)

    # gamma的折扣率它必须介于0和1之间。越大，折扣越小。这意味着学习，agent 更关心长期奖励。
    # 另一方面，gamma越小，折扣越大。这意味着我们的 agent 更关心短期奖励（最近的奶酪）。
    # epsilon探索率ϵ。即策略是以1−ϵ的概率选择当前最大价值的动作，以ϵ的概率随机选择新动作。
    agent = Agent(
        gamma=0.9,
        epsilon=0.03,
        batch_size=32,
        n_actions=5,
        eps_end=0.03,
        eps_dec=0.0001,
        input_dims=[13],
        lr=lr,
        fc1_dims=256,
        fc2_dims=128
    )

    if lr == 0.002:
        agent.load_model(
            f'model/batch{test_batch}/{bankuaicode}_{LTR}_{train_year}year_top4_train{train_year}TESToc'
        )
    elif lr != 0.002:
        agent.load_model(
            f'model/batch{test_batch}/{bankuaicode}_{LTR}_{train_year}year_top4_train{train_year}TESToc_{lr}'
        )

    funds, day_return, real_actions, eps_history, select_positive_counts, losss, date = [], [], [], [], [], [], []
    fixed_actions = []

    day_profit = 0
    done = False
    observation = env.reset()

    while not done:
        date.append(env.qid_date)

        if use_fixed_action:
            # lr == 0.002：固定每日选择结果，用于 T4 / Fig5 复现
            if int(env.qid_date) not in fixed_action_map:
                print("日期没匹配上：", env.qid_date)

            action = int(fixed_action_map.get(int(env.qid_date), 0))
            fixed_actions.append(action)

            observation_, reward, done, real_action, positive_count = env.step(action)

            # 固定 action 时，不训练 DQN
            loss = 0.0000

        else:
            # lr != 0.002：恢复原始 DQN 动态 action，用于 F3b 敏感性分析
            action1 = agent.choose_action(observation)
            action = action1
            fixed_actions.append(np.nan)

            observation_, reward, done, real_action, positive_count = env.step(action)

            agent.store_transition(observation, action, reward, observation_, done)
            loss = agent.learn()

        day_profit = env.day_profit
        fund = env.fund
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
        'fixed_action': fixed_actions,
        'real_action': real_actions,
        'select_positive_count': select_positive_counts
    }

    result_df = pd.DataFrame(result_data)

    # 默认不写入这个明细文件
    # result_df.to_excel(
    #     f'result/batch{test_batch}/{bankuaicode}_{LTR}_{train_year}year_train{train_year}_top4TESToc_{lr}xinxin.xlsx',
    #     index=False
    # )

    results_df = result_df[test_start <= result_df['qid_date']].copy()

    if args.write_excel:
        count_col_to_write = "fixed_action" if use_fixed_action else "real_action"

        T4ExcelWriter(scale=500).write(
            results_df,
            dapan_code=bankuaicode,
            target_col="LTR-DQN",
            value_col="funds",
            count_col=count_col_to_write,
            count_target_col="number of stocks"
        )

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