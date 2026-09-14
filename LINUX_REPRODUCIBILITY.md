# Linux 结果一致性说明

Linux 是下载到 Windows 不便时使用的应急兼容路径，不是 GitHub 在线验证环境。GitHub Actions 只允许使用 `windows-2022`。Linux 运行会由 `runtime_config.py` 自动读取 `parameters_linux.txt`，无需在命令行输入参数文件。

本项目的 Linux 路径分为两种依赖模式：

1. **锁定模式（默认）**：Python 3.9.13、NumPy 1.21.5、pandas 1.4.4、SciPy 1.10.1、scikit-learn 1.2.0、PyTorch 2.0.0+cpu 和 XGBoost 1.7.6 与 Windows 参考环境一致。入口脚本发现版本不符会停止。
2. **放宽模式（显式开启）**：设置 `LTR_DQN_RELAXED_RUNTIME=1` 后允许其他版本运行。该模式只保证流程可执行，不保证结果接近，运行清单会记录实际版本。

为减少跨平台差异，两套 Linux 脚本都会先断言当前系统为 Linux 且活动配置为 `linux-emergency-compatibility`，再设置 `PYTHONHASHSEED=0`、固定随机种子、CPU 单线程、`n_jobs=1`、`ATEN_CPU_CAPABILITY=default` 和 `MKL_CBWR=COMPATIBLE`。代码中的输入输出路径都相对于仓库根目录解析，不依赖 Windows 盘符或反斜杠。

即使锁定模式完全成功，也不能承诺逐位一致。原因包括：XGBoost/PyTorch wheel 的编译器和 CPU 指令集不同，Linux 与 Windows 的 BLAS/数学库实现不同，以及浮点加法顺序的差异。实际判断应比较 `results/combined/results_long.csv` 中 T4/T5/T7 的指标，关注 ARR、MDR、CR、SR、WR 的整体水平，而不是单个预测值或模型参数的逐位相同。

建议先运行：

```bash
python -m unittest discover -s tests -v
python -m compileall -q .
```

再运行 `bash run_linux.sh`。结果清单中的 `runtime_mode`、依赖版本、输入哈希和动作/检查点哈希用于确认两次运行是否具有可比的输入和环境。
