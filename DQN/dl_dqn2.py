import pandas as pd
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class Environment():
    def __init__(self, data, temp_data, start_date, end_date):
        self.data = data
        self.temp_data = temp_data

        self.data = self.data.fillna(0)
        self.temp_data = self.temp_data.fillna(0)
        self.data['qid_date'] = pd.to_numeric(self.data['qid_date'].str.replace('-', ''))
        self.data['trade_date'] = pd.to_numeric(self.data['trade_date'].str.replace('-', ''))
        # 提取需要归一化的列
        features_to_normalize = self.data.drop(columns=['qid_date', 'trade_date'])
        # 计算最小值和最大值
        min_vals = features_to_normalize.min()
        max_vals = features_to_normalize.max()
        # 进行 [-1, 1] 标准化处理
        normalized_features = 2 * (features_to_normalize - min_vals) / (max_vals - min_vals) - 1
        # 将归一化后的数据合并回原 DataFrame
        self.data = self.data[['qid_date', 'trade_date']].join(normalized_features)

        self.start_date = start_date
        self.end_date = end_date
        self.qid_date = 20220930

        self.buy_fee_rate = 0.0003
        self.sell_fee_rate = 0.0003 + 0.001

        self.init = 500_000_000
        self.fund = 500_000_000

        self.total_profit = 0
        self.day_profit = 0
        self.select_num = 0
        self.day_real_return = 0
        self.select_positive_count = 0

    def reset(self):
        self.qid_date = self.start_date

        self.init = 500_000_000  #初始资金
        self.fund = 500_000_000 #剩余资金

        self.total_profit = 0  #总收益
        self.day_profit = 0  #单日收益
        self.select_num = 0
        self.day_real_return = 0
        self.select_positive_count = 0

        observation = self.data[self.data['trade_date'] == self.qid_date].values.flatten().tolist()
        observation.append(self.day_real_return/0.10)
        observation.append((self.select_num-2)/2)
        observation.append((self.select_positive_count-2)/2)
        return (observation)

    def step(self, action_):
        #
        selected_df = self.temp_data[self.temp_data['qid_date'] == self.qid_date]
        # 基于 prediction 由高到低排序
        selected_df = selected_df.sort_values(by='prediction', ascending=False)
        # 选择前 action 个研报数据
        if action_ > len(selected_df):
            action_ = len(selected_df)
        if action_ > 0:
            selected_df = selected_df.head(action_)
            # self.day_real_return = selected_df['real_return'].mean()-(self.buy_fee_rate+self.sell_fee_rate)
            self.select_positive_count = (selected_df['real_return'] > 0).sum()

            capital_per_stock = self.fund / action_
            self.day_profit = 0
            for _, row in selected_df.iterrows():
                # 计算能买的股数（向下取整）
                shou_num = int(capital_per_stock / (100 * row['pclose']))
                buyfei = shou_num * 100 * row['pclose'] * self.buy_fee_rate
                shares_bought = int((capital_per_stock - self.buy_fee_rate) / (100 * row['pclose'])) * 100
                yu = capital_per_stock - buyfei - shares_bought * row['pclose']
                if shares_bought == 0:
                    print(f"Warning: Not enough capital to buy any shares of stock {row['stock_code']} on {self.qid_date}.")
                    yu = capital_per_stock
                # 计算卖出后的资金
                sell_value = shares_bought * row['close'] - shares_bought * row['close'] * (self.sell_fee_rate) + yu
                # 累加当日总收益
                self.day_profit += (sell_value - capital_per_stock)
            self.day_real_return = self.day_profit / self.fund

        else:
            self.day_real_return = 0.0
            self.select_positive_count = 0
            self.day_profit = 0.0
        self.select_num = action_
        # self.day_profit = self.fund * self.day_real_return
        self.fund = self.fund + self.day_profit
        self.total_profit += self.day_profit

        self.qid_date = self.data[self.data['qid_date'] == self.qid_date]['trade_date'].iloc[0]
        observation_ = self.data[self.data['trade_date'] == self.qid_date].values.flatten().tolist()
        observation_.append(self.day_real_return/0.10)
        observation_.append((self.select_num-2)/2)
        observation_.append((self.select_positive_count-2)/2)

        reward = self.day_real_return
        if self.qid_date == self.end_date:
            done = True
        else:
            done = False

        return (observation_, reward, done, action_, self.select_positive_count)


class DeepQNetwork(nn.Module):
    def __init__(self, lr, input_dims, fc1_dims, fc2_dims, n_actions):
        super(DeepQNetwork, self).__init__()
        self.lr = lr
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions

        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.fc3 = nn.Linear(self.fc2_dims, self.n_actions)
        # self.optimizer = torch.optim.SGD(self.parameters(), lr=lr, momentum=0.9)
        # self.optimizer = optim.RMSprop(self.parameters(), lr=lr, alpha=0.9)#设置优化器
        self.optimizer = optim.Adam(self.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08)
        self.loss = nn.MSELoss()

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        state.to(self.device)
        x1 = F.tanh(self.fc1(state.to(T.float32)))
        x2 = F.tanh(self.fc2(x1))
        # x = F.tanh(self.fc2(x))
        actions = self.fc3(x2)

        return actions


class Agent():
    # gamma的折扣率它必须介于0和1之间。越大，折扣越小。这意味着学习，agent 更关心长期奖励。另一方面，gamma越小，折扣越大。这意味着我们的 agent 更关心短期奖励（最近的奶酪）。
    # epsilon探索率ϵ。即策略是以1−ϵ的概率选择当前最大价值的动作，以ϵ的概率随机选择新动作。
    def __init__(self, gamma, epsilon, lr, input_dims, batch_size, n_actions=5,
                 max_mem_size=100, eps_end=0.03, eps_dec=0.0002, fc1_dims=256,fc2_dims=128):
        self.replace_target_iter = 8
        self.learn_step_counter = 0
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_min = eps_end
        self.eps_dec = eps_dec
        self.lr = lr
        self.n_actions = n_actions
        self.action_space = [i for i in range(n_actions)]
        self.mem_size = max_mem_size
        self.batch_size = batch_size
        self.mem_cntr = 0

        self.Q_eval = DeepQNetwork(self.lr, input_dims=input_dims, n_actions=self.n_actions,
                                   fc1_dims=fc1_dims, fc2_dims=fc2_dims)

        self.Q_target = DeepQNetwork(self.lr, input_dims=input_dims, n_actions=self.n_actions,
                                   fc1_dims=fc1_dims, fc2_dims=fc2_dims)

        self.state_memory = np.zeros((self.mem_size, *input_dims), dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *input_dims), dtype=np.float32)

        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=bool)

    def save_model(self, model_path):
        T.save(self.Q_eval, model_path)

    def load_model(self, model_path):
        self.Q_eval = T.load(model_path)

    # 存储记忆
    def store_transition(self, state, action, reward, state_, done):

        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.reward_memory[index] = reward
        self.action_memory[index] = action
        self.terminal_memory[index] = done

        self.mem_cntr += 1
        print("store_transition index:", index)

    # observation就是状态state
    def choose_action(self, observation):
        r = np.random.random()
        if r > self.epsilon:
            print('not random action')
            # 随机0-1，即1-epsilon的概率执行以下操作,最大价值操作
            state = T.tensor(observation).to(self.Q_eval.device)
            # 放到神经网络模型里面得到action的Q值vector
            actions = self.Q_eval.forward(state)
            action = T.argmax(actions).item()
        else:
            # epsilon概率执行随机动作
            action = np.random.choice(self.action_space)
            print("random action:", action)
        return action

    def _replace_target_params(self):
        # 复制网络参数
        print(self.Q_eval)
        print(self.Q_target)
        self.Q_target.load_state_dict(self.Q_eval.state_dict())

    # 从记忆中抽取batch进行学习
    def learn(self):
        # memory counter小于一个batch大小的时候直接return
        if self.mem_cntr < self.batch_size:
            print("learn:watching")
            return 0.0000

        if self.learn_step_counter % self.replace_target_iter == 0:
            self._replace_target_params()
            print('\ntarget params replaced\n')

        # 初始化梯度0
        self.Q_eval.optimizer.zero_grad()

        # 得到memory大小，不超过mem_size
        max_mem = min(self.mem_cntr, self.mem_size)

        # 随机生成一个batch的memory index，不可重复抽取
        batch = np.random.choice(max_mem, self.batch_size, replace=False)

        # int序列array，0~batch_size
        batch_index = np.arange(self.batch_size, dtype=np.int32)

        # 从state memory中抽取一个batch
        state_batch = T.tensor(self.state_memory[batch]).to(self.Q_eval.device)
        new_state_batch = T.tensor(self.new_state_memory[batch]).to(self.Q_eval.device)
        reward_batch = T.tensor(self.reward_memory[batch]).to(self.Q_eval.device)
        terminal_batch = T.tensor(self.terminal_memory[batch]).to(self.Q_eval.device)  # 存储是否结束的bool型变量

        # action_batch = T.tensor(self.action_memory[batch]).to(self.Q_eval.device)
        action_batch = self.action_memory[batch]

        # 第batch_index行，取action_batch列,对state_batch中的每一组输入，输出action对应的Q值,batchsize行，1列的Q值
        q_eval = self.Q_eval.forward(state_batch)[batch_index, action_batch]
        # q_next = self.Q_eval.forward(new_state_batch)
        q_next = self.Q_target(new_state_batch)
        q_next[terminal_batch] = 0.0  # 如果是最终状态，则将q值置为0
        q_target = reward_batch + self.gamma * T.max(q_next, dim=1)[0]
        loss = self.Q_eval.loss(q_target, q_eval).to(self.Q_eval.device)
        loss.backward()
        self.Q_eval.optimizer.step()

        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min \
            else self.eps_min
        self.learn_step_counter += 1
        return loss.item()