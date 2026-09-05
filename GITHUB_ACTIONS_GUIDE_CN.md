# 新 GitHub 账号一键复现指南

本指南适用于一个刚注册的 GitHub 账号。无需在本地安装 Python，也无需配置 GPU。只要把本仓库 Fork 到自己的账号，就可以使用 GitHub Actions 从仓库中的原始数据重新训练模型，并生成论文表格和图片。

在线复现分为两个独立任务：

- **1 - Results and Figures (CPU)**：生成 T3、T4、T5、T7、全部正文图，以及附录图 C1、C2、C3、C5。
- **2 - T6 and Figure C4 (CPU)**：单独生成 T6 和附录图 C4。

两个任务都固定使用 GitHub 提供的标准 Windows Server 2022 x64 CPU。这个复现版本不支持 Linux runner。无需创建或选择 runner，也无需选择 GPU。

## 一、准备新账号

1. 登录新的 GitHub 账号。
2. 打开原仓库：<https://github.com/whyhw-code/LTR-DQN>。
3. 确认页面显示的是 `main` 分支。

建议将复现仓库保持为 **Public（公开）**。本项目使用标准 Windows GitHub-hosted runner；公开仓库使用这种标准 runner 不收取 GitHub Actions 运行分钟费用。请勿把工作流改成 Linux、larger runner、GPU runner 或自托管付费设备。

## 二、复制仓库到新账号

1. 点击仓库页面右上角的 **Fork**。
2. 在 **Owner** 中选择你的新账号。
3. **Repository name** 保持为 `LTR-DQN`，也可以填写自己的名称。
4. 勾选 **Copy the main branch only**（只复制 main 分支）。
5. 点击 **Create fork**。
6. 等待页面跳转到新账号下的仓库，例如：

   ```text
   https://github.com/你的账号/LTR-DQN
   ```

Fork 会复制原始数据、脚本、环境锁定文件和两个在线复现工作流，不会复制以前的运行结果。

## 三、启用 Actions

1. 在新账号的 Fork 仓库中点击顶部的 **Actions**。
2. 如果页面显示 **I understand my workflows, go ahead and enable them**，点击该按钮。
3. 左侧应出现以下两个工作流：

   ```text
   1 - Results and Figures (CPU)
   2 - T6 and Figure C4 (CPU)
   ```

如果已经直接看到这两个名称，说明 Actions 已启用，不需要额外设置。

## 四、运行 Results 和正文图

1. 在 Actions 页面左侧点击 **1 - Results and Figures (CPU)**。
2. 点击右侧的 **Run workflow**。
3. 如果出现分支选择框，保持 **Branch: main**。
4. 再点击绿色的 **Run workflow** 确认启动。
5. 页面出现新的运行记录后，点击该记录查看进度。

这个任务会从原始数据重新训练模型，然后生成：

```text
results/combined/results.xlsx       T3、T4、T5、T7 总工作簿
results/combined/                   论文格式 CSV 和运行清单
results/figures/                    全部正文图及审计 CSV
results/appendix_figures/           附录图 C1、C2、C3、C5 及审计 CSV
```

这不是读取已上传的模型或历史结果：每次运行都会从仓库跟踪的原始数据开始重新训练。

## 五、单独运行 T6 和附录图 C4

1. 返回仓库的 **Actions** 页面。
2. 在左侧点击 **2 - T6 and Figure C4 (CPU)**。
3. 点击 **Run workflow**。
4. 如果出现分支选择框，保持 **Branch: main**。
5. 点击绿色的 **Run workflow** 确认启动。

这个任务独立重新训练三年期模型并执行 T6 的 20 次抽样，生成：

```text
results/T6/T6.xlsx                  T6 工作簿
results/appendix_figures/           附录图 C4 及审计 CSV
temp/t6_runs/t6_raw.csv             T6 原始抽样审计结果
temp/t6_runs/t6_manifest.json       T6 运行清单
```

T6 的 100% 结果来自同一次运行中新评估的 T4 模型，不依赖外部中间结果。

## 六、判断是否完成

- 黄色圆点：正在排队或运行，请继续等待。
- 绿色对勾：运行成功，可以下载结果。
- 红色叉号：运行失败。点开失败记录，再点开带红叉的步骤查看错误信息。

浏览器页面可以关闭，GitHub 会继续运行。两个任务相互独立，可以先运行 Results，完成后再运行 T6，便于分别查看结果和失败原因。

## 七、下载结果

1. 点击带绿色对勾的运行记录。
2. 在运行详情页向下找到 **Artifacts** 区域。
3. 点击对应文件包名称下载 ZIP：

   ```text
   ltr-dqn-results-and-figures-运行编号
   ltr-dqn-t6-and-c4-运行编号
   ```

4. 解压 ZIP 后，按上面的 `results/` 路径查找工作簿和图片。

Artifacts 保留 14 天。超过期限后需要重新运行工作流。失败的运行通常不会生成完整的最终结果包。

## 八、常见问题

### Actions 页面没有工作流

确认你打开的是自己 Fork 后的仓库，而不是账号设置页或原仓库的某个文件页面。然后进入仓库顶部的 **Actions**，按第三节启用工作流。

### 看不到 Run workflow 按钮

确认当前账号对该仓库有写入权限，并确认工作流位于默认分支 `main`。刷新页面后重新选择左侧的工作流名称。

### GitHub 要求选择分支

选择 `main`。不需要选择其他分支。

### GitHub 要求选择 runner

本仓库的 runner 已在工作流中固定为 `windows-2022`，正常启动页面不会要求选择。不要进入组织的 **Settings > Actions > Runners** 创建机器；直接在仓库的 **Actions** 页面运行即可。

### 环境安装失败

打开失败记录，展开 **Install locked environment**，保存完整错误信息。工作流锁定 Windows Server 2022 x64、Python 3.9.13、pip 24.1.2 和实验依赖版本，安装失败时不要随意升级包版本，否则可能改变结果。2026 年 9 月 5 日以前 Fork 的 Linux 版本如果显示 `xgboost 1.7.6 is not supported on this platform`，请先按下方“Fork 后原仓库更新了”同步 `main`，再重新运行 Windows 版本。

### Fork 后原仓库更新了

进入你 Fork 的仓库首页。如果分支栏附近显示 **Sync fork**，点击它，再点击 **Update branch**，把原仓库的最新 `main` 同步到自己的 `main`。同步完成后回到 Actions 页面重新运行；不要继续点击旧失败记录中的 **Re-run jobs**，因为旧记录仍使用旧提交。

### 运行成功但找不到结果

结果不会提交到仓库文件列表中，而是在该次运行详情页底部的 **Artifacts** 中下载。

### 是否会收费

当 Fork 保持公开，并且工作流仍使用标准 `windows-2022` GitHub-hosted runner 时，GitHub Actions 的标准 runner 运行分钟不收费。存储和使用政策仍以 GitHub 当前规则为准。不要改用 larger runner 或 GPU runner，它们可能单独计费。

## 九、复现检查清单

开始前确认：

- 仓库是从 `whyhw-code/LTR-DQN` Fork 得到的。
- 默认分支是 `main`。
- 仓库保持 Public。
- 没有修改 `requirements-lock.txt`、原始数据或工作流。
- Results 和 T6 分别运行各自的工作流。
- 下载的是带绿色对勾那次运行的 Artifacts。

GitHub 官方说明：

- [手动运行工作流](https://docs.github.com/en/actions/how-tos/manage-workflow-runs/manually-running-a-workflow)
- [启用或禁用工作流](https://docs.github.com/en/actions/how-tos/manage-workflow-runs/disabling-and-enabling-a-workflow)
- [下载工作流 Artifacts](https://docs.github.com/en/actions/how-tos/manage-workflow-runs/downloading-workflow-artifacts)
- [GitHub Actions 计费说明](https://docs.github.com/en/billing/concepts/product-billing/github-actions)
