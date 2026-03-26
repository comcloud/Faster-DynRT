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
  --tensor-root /root/autodl-tmp/datasets/msd/tensor \
  --source prepared \
  --size 224
```

## 本地执行 + 打包上传

本地执行 10 组：

```bash
export AGENT_BACKEND=openai_api
export BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
export KEY=你的key
export MODEL=gemini:gemini-3.1-flash-lite-preview
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

## 4. train.py 的兼容变更

`train.py` 现在支持：

```bash
python train.py config/DynRT.json
```

如果不传参数，仍会使用默认 `config/DynRT.json`（也支持环境变量 `DYNRT_CONFIG` 作为默认值）。
