# XVLA-TCLA

[English](README.md)

XVLA-TCLA 将 TCLA v3 动作自编码器集成到 [X-VLA](https://github.com/2toinf/X-VLA)，
支持 RoboTwin 2.0 和 Franka 的训练与推理。本项目是衍生项目，并非 X-VLA 官方版本。
视觉语言骨干和软提示策略 Transformer 来自 X-VLA；TCLA 增加了动作压缩、轨迹条件解码，
以及可在本项目内独立运行的 AE 训练流程。

## 三种训练模式

| 模式 | 训练目标 | Autoencoder 状态 | 推理输出 |
| --- | --- | --- | --- |
| `raw` | 绝对关节动作及夹爪动作 | 不加载，不需要 AE 权重或统计文件 | 每次请求返回一个物理动作 |
| `latent` | AE 潜变量，可附加解码动作损失 | 加载预训练 AE，全程冻结 | 潜变量经 AE 解码为物理动作 |
| `latent_joint` | 策略损失，以及 AE 重建、进度、动力学损失 | AE 联合训练，EMA 目标编码器保持冻结 | 使用 Policy checkpoint 内保存的联合训练 AE |

RoboTwin 动作为 14 维：左臂 6 维、左夹爪、右臂 6 维、右夹爪。
Franka 动作为 8 维：7 个关节及夹爪。raw 保留 X-VLA 的关节 MSE 和夹爪 BCE 损失，
夹爪训练标签须在 `[0, 1]` 内。两种 latent 模式均使用 AE 的动作均值和标准差归一化。

## 安装环境

```bash
conda create -n xvla-tcla python=3.10 -y
conda activate xvla-tcla
pip install -r requirements.txt
```

请安装与 CUDA 驱动匹配的 PyTorch/torchvision。训练启动脚本需要 Bash，以下命令均在本项目
目录执行。RoboTwin 仿真环境和 Franka 控制器使用各自的运行环境。
AE 不再依赖旁边的 TCLA 仓库或另一个 Python 环境。

## 统一路径配置

公共默认值位于 [configs/paths.yaml](configs/paths.yaml)。机器相关路径建议写入被 Git 忽略的
`configs/paths.local.yaml`，只需填写要覆盖的字段：

```yaml
paths:
  data:
    robotwin: data/robotwin
    franka: data/franka
  pretrained_model: 2toINF/X-VLA-Pt
  language_model: sentence-transformers/all-MiniLM-L6-v2
  action_ae:
    robotwin: results/robotwin_ae/selected_run
    franka: results/franka_ae/selected_run
  policy_checkpoint: results/robotwin_policy/raw/ckpt-50000
  robotwin: external/RoboTwin
  eval_output: results/evaluation
```

文件相对路径统一以项目根目录为基准。模型字段可以填写 Hugging Face ID 或本地目录；
本地相对模型路径以 `./` 开头，例如 `./checkpoints/X-VLA-Pt`。
设置 `XVLA_TCLA_PATHS_CONFIG` 可使用其他私有路径配置文件。
输出目录也可通过 `paths.output` 配置。运行 `python configuration.py` 查看解析后的路径。

## 数据格式

每个任务一个目录，其中包含 `episode_0.hdf5`、`episode_1.hdf5` 等文件。

| HDF5 字段 | RoboTwin | Franka |
| --- | --- | --- |
| `action` | `[T, 14]`，绝对关节及夹爪动作 | `[T, 8]`，绝对关节及夹爪动作 |
| `observations/qpos` | `[T, 14]` | `[T, 8]` |
| `observations/images/<camera>` | RGB uint8 `[T, H, W, 3]` | RGB uint8 `[T, H, W, 3]` |
| 相机顺序 | `cam_high`、`cam_left_wrist`、`cam_right_wrist` | `front_cam`、`wrist_cam` |
| AE/latent 的 `task_emb` | 必填，`[384]` 或 `[1, 384]` | 按 AE 配置从任务提示词生成 |

AE 训练不读取图像，raw 训练不读取 `task_emb`。AE 将 `episode_0` 留作验证集。
Franka 的 AE 和联合训练要求每条 episode 至少包含两个完整动作块，默认是 64 步。
Policy 配置中设置 `data.task_names=null` 可以自动发现所有任务目录，也适用于 RoboTwin
50 任务数据。AE 的任务列表单独通过 `task_name` 设置。AE 与 Policy 的任务名和语言嵌入
必须保持一致。

## 独立训练 Autoencoder

AE 配置位于 [RoboTwin](configs/ae/robotwin.yaml) 和 [Franka](configs/ae/franka.yaml)，
来自 TCLA 的 `train_version3.sh` 配置。默认使用 32 步动作块、48 个 DCT 频率分量、
8 个 16 维潜变量 token、EMA 目标编码器、语言条件 Perceiver 进度模型及恒定学习率。

```bash
bash scripts/train_ae.sh robotwin 0
bash scripts/train_ae.sh franka 0
bash scripts/train_ae.sh robotwin 0,1 num_freqs=24 trainer.max_train_steps=160000
```

在 `framework` 中配置模型维度和损失权重。保留的组件类型是 `tf_seq_cnn`、
`tf_attn_w_nonScalarPhase`、`direct_flow_w_nonScalarPhase` 和 `perceiver`。
用 `num_freqs` 修改频率分量数，用 `n_action_steps` 修改动作块长度，保证数据和模型同步。
Franka 默认使用 MiniLM。显式设置 `datasets.action_data.dataset.task_emb_source=hash`
可训练 hash 条件 AE；使用已经训练好的 checkpoint 时，不要更换语言嵌入来源。

带时间戳的 AE 运行目录会保存 `config.yaml`、`dataset_statistics.json`、
`checkpoints/steps_<step>_pytorch_model.pt`、验证 CSV 和潜空间可视化。
AE checkpoint 包含优化器及学习率调度器状态，恢复训练示例：

```bash
python train_ae.py --config_yaml configs/ae/robotwin.yaml \
  trainer.is_resume=true \
  trainer.resume_checkpoint=results/robotwin_ae/selected_run/checkpoints/steps_160000_pytorch_model.pt
```

训练完成后，将 `paths.action_ae.<platform>` 指向实际运行目录。
Policy 通过 `action_ae.step` 选择权重步数，默认 `160000`，并自动读取 AE 保存的模型维度、
频率设置和归一化参数。当前 Policy 接入使用绝对动作，潜变量宽度须不小于物理动作维度。

## 训练 Policy

[base.yaml](configs/policy/base.yaml) 保存公共优化器和损失设置。
[robotwin.yaml](configs/policy/robotwin.yaml) 和 [franka.yaml](configs/policy/franka.yaml)
指定数据、任务、相机及 domain ID。

```bash
bash scripts/train_policy.sh robotwin raw 0
bash scripts/train_policy.sh robotwin latent 0
bash scripts/train_policy.sh robotwin latent_joint 0
bash scripts/train_policy.sh franka raw 0
bash scripts/train_policy.sh franka latent 0
bash scripts/train_policy.sh franka latent_joint 0
bash scripts/train_policy.sh robotwin latent 0,1 action_ae.step=160000 training.batch_size=16
```

`training.batch_size` 是每张 GPU 的 batch size。`MAIN_PROCESS_PORT` 设置分布式端口；
`MIXED_PRECISION=bf16` 开启 Policy 混合精度，AE 运算仍保持 fp32。
`training.freeze_steps` 控制 VLA 骨干训练初期的学习率冻结，不控制 AE：
`latent` 全程冻结 AE，`latent_joint` 全程允许 AE 更新。
`latent.decoded_loss_coef=0` 可关闭解码动作训练损失。
联合训练 AE 的学习率与辅助损失权重位于 `joint_ae`。

```bash
python train.py --config configs/policy/franka.yaml --mode latent --print_config
python train.py --config configs/policy/robotwin.yaml --mode raw training.iters=50000
```

Policy checkpoint 默认位于 `results/<platform>_policy/<mode>/ckpt-<step>`，包含模型、
processor 及 `policy.json`。latent checkpoint 还包含 `action_ae_config.yaml`、
`action_ae.pt`、`dataset_statistics.json`、`qpos_stats.json` 和 `task_embeddings.json`。
部署时迁移整个 checkpoint 目录即可。Policy checkpoint 是推理导出文件，不含优化器恢复状态。

## 推理与评估

```bash
python serve.py --checkpoint results/robotwin_policy/latent/ckpt-50000 --port 8000
python -m evaluation.robotwin --task adjust_bottle --episodes 50 --save_video
```

服务端从 checkpoint 自动识别 raw、latent 或联合训练模式。不传 `--checkpoint` 时使用
`paths.policy_checkpoint`。`--steps` 设置 VLA 去噪步数，`--execute_steps` 设置每次重新
规划前执行的动作数量，默认执行完整动作块。
在 RoboTwin 仿真环境中从本项目目录启动评估命令；它从 `paths.robotwin` 加载仿真器，
将 episode 结果和成功率写入 `paths.eval_output`。`--task all` 评估 checkpoint 中记录的
全部任务。raw 和 latent 均向 RoboTwin 发送 14 维关节指令。

Franka 使用同一个 `serve.py` 加载 Franka checkpoint，在现有控制器中调用
[FrankaPolicyClient](inference/franka.py)：

```python
from inference.franka import FrankaPolicyClient

policy = FrankaPolicyClient("http://127.0.0.1:8000")
policy.reset("260830_bowl_stack")
action = policy.step(qpos, front_rgb, wrist_rgb)
```

`qpos` 是当前 7 个关节位置及夹爪状态，相机输入为 RGB uint8 数组。
`action` 是与训练数据单位一致的 8 维绝对动作目标，需要接到现有控制器的指令接口；
本仓库不包含厂商专用 Franka 硬件驱动。每个 episode 开始时调用 `reset`，随后每次请求
执行一个返回动作。latent 历史记录已下发的动作目标。每个独立控制的机器人使用一个服务进程。

HTTP 接口包括 `GET /info`、`POST /reset` 和 `POST /act`。
数组采用标准 JSON 数组；`/act` 接收 `qpos` 及按 checkpoint 相机顺序排列的
`image0`、`image1` 等字段，返回 `{"action": [...]}`。
同一任务各 episode 的嵌入一致时，会保存到 checkpoint；未见过的 RoboTwin 任务或逐 episode
变化的嵌入，需要在 `/reset` 中传入匹配的 `task_emb`。

## 致谢、引用与许可

原始 X-VLA 架构和预训练模型的贡献属于 [2toINF 及 X-VLA 作者](https://github.com/2toinf/X-VLA)。
请参阅 [论文](https://arxiv.org/abs/2510.10274) 和
[项目主页](https://thu-air-dream.github.io/X-VLA/)。研究中使用本衍生项目时，请引用 X-VLA，
并单独说明 TCLA 集成部分：

```bibtex
@article{zheng2025x,
  title   = {X-VLA: Soft-Prompted Transformer as Scalable Cross-Embodiment Vision-Language-Action Model},
  author  = {Zheng, Jinliang and Li, Jianxiong and Wang, Zhihao and Liu, Dongxiu and Kang, Xirui
             and Feng, Yuchun and Zheng, Yinan and Zou, Jiayin and Chen, Yilun and Zeng, Jia and others},
  journal = {arXiv preprint arXiv:2510.10274},
  year    = {2025}
}
```

保留原 [Apache 2.0 许可证](LICENSE)、[NOTICE](NOTICE) 和
[第三方说明](THIRD_PARTY_NOTICES.md)，以及 Florence-2 源码中的版权声明。
权重、数据、仿真器和机器人驱动遵循各自的许可。
私有路径配置、数据、权重和生成的输出由 `.gitignore` 排除。
