# DynRT 自动化改模与实验

这个目录提供一个最小可用框架：按计划文件自动调用外部智能体改模型，然后自动训练并汇总结果。

## 1. 先准备计划文件

复制模板并编辑：

```bash
cp automation/plan.example.json automation/plan.local.json
```

关键字段：

- `base_config`：DynRT 基础配置文件。
- `agent.command`：外部智能体命令模板。
  - 可用占位符：`{workspace}`、`{instruction_file}`、`{instruction}`、`{task_name}`。
  - 留空表示不调用智能体，只跑自动实验。
- `agent_tasks`：智能体任务列表，实验可通过 `agent_task` 引用。
- `experiments`：实验列表，支持：
  - `config_overrides`：点路径覆盖配置，例如 `opt.total_epoch`。
  - `env`：实验环境变量（比如 `CUDA_VISIBLE_DEVICES`）。
  - `train_args`：额外透传给 `train.py` 的参数。

## 2. 运行

先 dry-run 看命令是否正确：

```bash
python automation/auto_agent_experiment.py --plan automation/plan.local.json --dry-run
```

正式跑：

```bash
python automation/auto_agent_experiment.py --plan automation/plan.local.json
```

### 智能体后端切换（我已默认定好）

我把调用策略定成“适配器 + 环境变量”：

- 固定命令：`python automation/agent_adapter.py ...`
- `AGENT_BACKEND=noop`：不真正改代码（用于本地跑通流程）
- `AGENT_BACKEND=shell`：调用你指定的真实智能体命令模板
- `AGENT_BACKEND=openai_api`：调用 OpenAI 模型直接改目标文件（需要 `OPENAI_API_KEY`）

示例（远程服务器上启用真实智能体）：

```bash
export AGENT_BACKEND=shell
export AGENT_SHELL_TEMPLATE='codex exec --cwd {workspace} --instruction-file {instruction_file}'
python automation/auto_agent_experiment.py --plan automation/plan.module10.json
```

如果你用的不是 `codex exec`，只需要替换 `AGENT_SHELL_TEMPLATE`。

示例（直接用 OpenAI 作为真实智能体）：

```bash
pip install openai
export AGENT_BACKEND=openai_api
export OPENAI_API_KEY=你的key
export OPENAI_MODEL=gpt-5.4-mini
python automation/auto_agent_experiment.py --plan automation/plan.module10.remote.json
```

若使用 Gemini OpenAI 兼容网关，可设置：

```bash
export AGENT_BACKEND=openai_api
export BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
export KEY=你的key
export MODEL=gemini:gemini-3.1-flash-lite-preview
```

## 3. 结果输出

每次运行会生成：

- `auto_runs/<run_id>/summary.json`
- `auto_runs/<run_id>/summary.csv`
- `auto_runs/<run_id>/summary.md`

同时每个实验子目录包含：

- `config.json`：该实验实际使用的配置
- `agent.log`：智能体执行日志（如果启用）
- `train.log`：训练过程日志

脚本会自动尝试从新产生的 `exp/<time>/log.txt` 解析 `valid/test F1` 并写入汇总。
如果训练侧提供结构化指标文件（`metrics_history.json` / `best_model_info.json`），会优先记录：

- `acc`
- `micro-f1`
- `macro-f1`
- 当前实验的最优 checkpoint

并在每次运行目录下维护 `best_model.json`：
只有当出现更优模型时才更新，直到被下一个更好模型替换。

## 额外：10 组模块实验计划

已提供现成计划：`automation/plan.module10.json`

- 1 组 baseline
- 9 组模块改造（先不调超参数）

远程服务器推荐用：`automation/plan.module10.remote.json`
它默认读取 `config/DynRT.remote.template.json`（已把本地绝对路径替换成服务器路径模板）。

本地推荐用：`automation/plan.module10.msd.local.json`
它默认读取 `config/DynRT.msd.local.template.json`，并锁定 `prepared`。

## 数据说明（图文讽刺任务）

当前代码已支持三种数据源：`bully` / `prepared` / `msd2`，通过以下字段切换：

- `opt.dataloader.loaders.text.source`
- `opt.dataloader.loaders.img.source`
- `opt.dataloader.loaders.label.source`

本轮请固定使用：`prepared`（MSD 图文讽刺检测主数据集）。
不要使用 `bully`（仇恨检测）或 `MSD2` 作为当前主实验输入。

图片加载逻辑是：

1. 优先读 `transform_image` 目录下的 `.npy`
2. 若不存在则从 `image_root` 读取原图并自动转成 `.npy`

如果你要先离线预处理全部图片（推荐），可运行：

```bash
python automation/prepare_bully_images.py \
  --project-root . \
  --image-root /Users/rayss/Public/读研经历/论文/ironyDetection/imageVector2 \
  --tensor-root /root/autodl-tmp/ray/data/datasets/msd/tensor \
  --source prepared \
  --size 224
```

## 本地执行 + 打包上传

本地执行 10 组：

```bash
export AGENT_BACKEND=openai_api
export BASE_URL=http://43.139.186.69:8088/v1
export KEY=any-string
export MODEL=Qwen/Qwen3-235B-A22B-Instruct-2507
python automation/auto_agent_experiment.py --plan automation/plan.module10.msd.local.json
```

打包结果：

```bash
python automation/package_results.py \
  --run-root auto_runs/module10_msd_local \
  --output auto_runs/module10_msd_local.tar.gz
```

上传到远程：

```bash
rsync -az -e 'ssh -p 14722' auto_runs/module10_msd_local.tar.gz \
  root@connect.westb.seetacloud.com:/root/autodl-tmp/
```

## 自动闭环（效果差时继续改）

已提供自改进调度器：`automation/auto_self_improve.py`

- 先跑 baseline
- 自动尝试模型改造候选
- 若提升不足，再自动尝试参数候选
- 输出 `self_improve_summary.json` 记录每轮结果

示例：

```bash
python automation/auto_self_improve.py \
  --strategy automation/self_improve.strategy.msd.local.json
```

## 后台智能体控制器

已提供常驻控制器：`automation/background_controller.py`

它和一次性批跑不同，会在后台持续做这些事：

- 轮询训练日志和进程状态
- 遇到 `Traceback`、`RuntimeError`、卡死超时等情况时主动停训
- 调用智能体按失败日志修补目标文件
- 自动重试当前实验
- 如果新结果没有超过当前最优，会把代码恢复到已接受的最好版本
- 若指标进入平台期（波动很小）且当前效果仍低，会自动触发一次“模型结构重构”任务
- 持续写出状态和决策记录，便于断线后恢复观察

策略模板：

`automation/background_controller.strategy.example.json`

平台期配置项（在策略文件里）：

- `plateau.window`：用最近多少次结果判定平台期
- `plateau.mode`：`absolute`（绝对差）或 `relative`（相对变化率）
- `plateau.epsilon`：平台期阈值；当 `mode=relative` 时，`0.01` 表示变化率低于 1%
- `plateau.poor_metric_threshold`：若最佳指标仍低于该阈值，触发模型重构
- `plateau.target_files`：平台期重构允许修改的模型文件
- `post_train_retrain.*`：每次训练完成后判定是否“记录本次结果并触发改模型重训”
  - `change_ratio_threshold=0.01` 表示相邻两次指标变化率低于 1%
  - `poor_metric_threshold` 表示当前指标仍低于可接受阈值
- `agent_guidance_file`：每次智能体改模型前自动注入的全局研究指导（例如 `automation/agent_research_guidance.zh.md`）
- `agent_skill_files`：可附加多个 skill 文件（如 `SKILL.md` / `openai.yaml`），会在每次改模指令前自动注入
- `agent_skill_max_chars`：skill 注入的最大字符预算，防止提示词过长
- `retention.*`：自动留存策略，控制磁盘增长
  - `keep_exp_last`：只保留最近 N 个训练 exp 目录（外加当前最佳）
  - `prune_checkpoints=true`：每个 exp 只保留 best checkpoint（默认 `model_best.pth.tar`）
  - `keep_attempt_dirs`：只保留最近 N 个控制器 attempt 目录
  - `keep_train_log_lines` / `keep_agent_log_lines`：日志只保留最新若干行
  - `keep_events_lines`：`events.jsonl` 只保留最新若干行
  - `keep_results_last`：`results.json` 只保留最近若干条尝试记录
  - `drop_instruction_files` / `drop_agent_logs`：是否删除指令文件或 agent 日志

常用输出：

- `controller_state.json`：当前运行状态、当前实验、PID、最佳结果
- `results.json`：每次尝试的训练结果
- `events.jsonl`：按时间顺序的控制器决策日志

远程后台运行示例：

```bash
cd /root/autodl-tmp/ray/projects/dynrt_bridge_main
source /root/miniconda3/etc/profile.d/conda.sh
conda activate dynrt39_fresh
export AGENT_BACKEND=openai_api
export BASE_URL=http://43.139.186.69:8088/v1
export KEY=any-string
export MODEL=Qwen/Qwen3-235B-A22B-Instruct-2507
nohup python automation/background_controller.py \
  --strategy automation/background_controller.strategy.example.json \
  > auto_runs/background_controller.launch.log 2>&1 < /dev/null &
```

## 4. train.py 的兼容变更

`train.py` 现在支持：

```bash
python train.py config/DynRT.json
```

如果不传参数，仍会使用默认 `config/DynRT.json`（也支持环境变量 `DYNRT_CONFIG` 作为默认值）。

另外新增：

- `opt.save_epoch_checkpoints`（默认 `true`）：是否保存按 epoch 的周期 checkpoint。
- 当设置为 `false` 时，仅保留 `model_best.pth.tar`，可显著减少磁盘占用。
