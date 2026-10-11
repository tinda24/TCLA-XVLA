# XVLA-TCLA

[English](README.md)

XVLA-TCLA 将 TCLA 动作自编码器集成到 [X-VLA](https://github.com/2toinf/X-VLA)，
支持 RoboTwin 和 Franka 的训练与推理。本项目是基于 X-VLA 的独立衍生项目。
视觉语言骨干和软提示策略 Transformer 来自 X-VLA；TCLA 增加了动作压缩、轨迹条件解码，
以及可在本项目内独立运行的 AE 训练流程。

## 两种训练模式

| 模式 | 训练目标 | Autoencoder 状态 | 推理输出 |
| --- | --- | --- | --- |
| `raw` | 绝对关节动作及夹爪动作 | 不加载，不需要 AE 权重或统计文件 | 每次请求返回一个物理动作 |
| `latent` | AE 潜变量，可附加解码动作损失 | 加载预训练 AE，全程冻结 | 潜变量经 AE 解码为物理动作 |

RoboTwin 动作为 14 维：左臂 6 维、左夹爪、右臂 6 维、右夹爪。
Franka 动作为 8 维：7 个关节及夹爪。raw 保留 X-VLA 的关节 MSE 和夹爪 BCE 损失，
夹爪训练标签须在 `[0, 1]` 内。latent 模式使用 AE 的动作均值和标准差归一化。

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
Franka 的 AE 训练要求每条 episode 至少包含两个完整动作块，默认是 64 步。
Policy 配置中设置 `data.task_names=null` 可以自动发现所有任务目录，也适用于 RoboTwin
50 任务数据。AE 的任务列表单独通过 `task_name` 设置。AE 与 Policy 的任务名和语言嵌入
必须保持一致。

## 独立训练 Autoencoder

AE 配置位于 [RoboTwin](configs/ae/robotwin.yaml) 和 [Franka](configs/ae/franka.yaml)，
采用 TCLA 的自编码器训练配置。默认使用 32 步动作块、48 个 DCT 频率分量、
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
`checkpoints/steps_<step>_pytorch_model.pt` 和验证 CSV。
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
bash scripts/train_policy.sh franka raw 0
bash scripts/train_policy.sh franka latent 0
bash scripts/train_policy.sh robotwin latent 0,1 action_ae.step=160000 training.batch_size=16
```

`training.batch_size` 是每张 GPU 的 batch size。`MAIN_PROCESS_PORT` 设置分布式端口；
`MIXED_PRECISION=bf16` 开启 Policy 混合精度，AE 运算仍保持 fp32。
`training.freeze_steps` 控制 VLA 骨干训练初期的学习率冻结，不控制 AE：
`latent` 全程冻结 AE。
`latent.decoded_loss_coef=0` 可关闭解码动作训练损失。

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

服务端从 checkpoint 自动识别 raw 或 latent 模式。不传 `--checkpoint` 时使用
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

## 与上游 X-VLA Raw 的差异

本节只比较 RoboTwin `raw` 与 [X-VLA](https://github.com/2toinf/X-VLA) 的
RoboTwin2 微调、推理。对照对象是仓库 README 中的微调命令、`train.py`、
`datasets/domain_handler/simulations.py` 的 `RobotWin2Handler`，以及
`evaluation/robotwin-2.0/client.py`。预训练权重 `X-VLA-Pt` 的配置是
`action_mode=ee6d`、`num_actions=30`。

以下项与上游 README 微调示例一致：每卡 batch 16、学习率 `1e-4`、
`learning_coef=0.1`、50000 iter、冻结 1000 step、warmup 2000 step、
AdamW `(0.9, 0.95)`、weight decay 0、梯度裁剪 1.0、关闭 cosine。
domain id 对齐。上游 `robotwin2_abs_ee` 和 `robotwin2_clean` 共用 `RobotWin2Handler`，在 `datasets/domain_config.py` 里都映射为 6；域名只选择数据列表，模型使用的是这个整数。它从 `X-VLA-Pt` 的 30 行 soft prompt 中取出第 6 行，并选择动作编码器、动作解码器里 `DomainAwareLinear` 的第 6 组权重。本仓库没有域名表，`configs/policy/robotwin.yaml` 直接写 `domain_id: 6`，raw 和 latent 都用这一行。三路相机是 head、左腕、右腕。图像增强是 224 双三次缩放、
亮度/对比度/饱和度 0.2 抖动和 ImageNet 归一化。去噪步数默认 10。
评测配置名是 `demo_clean`，种子从 `2000 * (seed + 1)` 起。
`train.py` 源码默认值是 `learning_coef=1.0`、`iters=1000000`，和 README 示例不同；
本仓库跟随 README 示例。

### 训练

| 项 | 上游 X-VLA | 本仓库 raw |
| --- | --- | --- |
| 动作 | 绝对末端位姿，20 维：左右各 xyz(3) + rot6d(6) + gripper(1) | 绝对关节，14 维：左右各 6 关节 + 夹爪 |
| 动作头 | 继续训练预训练的 20 维 ee6d 头 | raw 和 latent 都在加载后重新初始化动作头，见下表 |
| 时间窗 | `num_actions=30`。未来最多 1 秒、按 30 Hz 插值采样；当前帧作本体感觉，其后 30 步作动作 | `chunk_size=32`。从当前帧起连续取 32 行，含当前帧动作；末尾重复最后一帧补齐 |
| 夹爪标签 | 末端夹爪从“1=张开”转为“1=闭合”后再做 BCE | 数据里关节夹爪已在 `[0, 1]`，通道 6 和 13 做 BCE |
| 任务规模 | RoboTwin2：50 个任务，每任务 50 条 | `configs/policy/robotwin.yaml`：10 个任务，每任务 50 条 |
| 精度 | 启动命令使用 `--mixed_precision bf16` | `scripts/train_policy.sh` 默认 `MIXED_PRECISION=no` |
| 存盘间隔 | `train.py` 默认 50000 step | `configs/policy/base.yaml` 为 10000 step |

训练语言读取 HDF5 的 `task_name`，例如 `adjust bottle`。这与上游评测把任务名下划线换成空格后的字符串相同。

`X-VLA-Pt` 的动作头是 20 维 ee6d：`action_encoder` 和 `action_decoder` 都是 `DomainAwareLinear`。本仓库两种模式都用 `ignore_mismatched_sizes=True` 加载，因为输入输出维度变了，然后用同一套初始化盖掉这两层：`fc.weight` 为 Xavier uniform，`bias.weight` 为 0。这么做是为了公平：raw 和 latent 都不继承预训练 ee6d 动作头，比较时起点相同。参数只计这两个 `DomainAwareLinear`。分母是对应配置下的整个 XVLA，包括 Florence 编码器、Transformer 和 soft prompt，不含已删除的语言 decoder 与 `lm_head`。预训练 20 维头是 2,857,560 / 879,738,545（0.325%）。换头之后，RoboTwin raw 为 2,304,420 / 879,185,405（0.262%），Franka raw 为 1,751,280 / 878,632,265（0.199%），latent 为 2,488,800 / 879,369,785（0.283%）。VLM 和 soft prompt 仍沿用预训练权重。损失函数模块本身没有预训练参数。

| | raw | latent |
| --- | --- | --- |
| 函数 | `train_raw.py` 的 `reinit_joint_action_head` | `train_latent.py` 的 `reinit_latent_action_head` |
| `action_mode` | RoboTwin `joint`，Franka `joint8` | `auto` |
| 动作维度 | RoboTwin 14，Franka 8 | AE 的 `latent_action_dim`，默认 16 |
| token 数 | `chunk_size=32` | `chunk_size / downsample_rate`，默认 8 |
| 预测对象 | 物理关节和夹爪 | AE 潜变量；可选再解码成物理动作 |
| 动作头参数 | RoboTwin 2,304,420，占该配置总参数 879,185,405 的 0.262%；Franka 1,751,280，占 878,632,265 的 0.199% | 2,488,800，占该配置总参数 879,369,785 的 0.283% |

### 推理

| 项 | 上游 X-VLA | 本仓库 raw |
| --- | --- | --- |
| 下发指令 | 20 维末端动作转成位置、四元数和夹爪后，`action_type='ee'` | 14 维关节目标，`action_type='qpos'` |
| 语言 | `task_name.replace('_', ' ')`，例如 `adjust bottle` | `evaluation/robotwin.py` 只把目录名传给 `reset`，服务端使用 `adjust_bottle` |
| 重规划 | 外层最多查询 10 次，每次执行完整的 30 步末端动作 | 服务端缓存动作块，每次 `/act` 返回一步；默认块长 32，评测最多 3000 个环境步 |
| 每任务条数 | `eval_robotwin.sh` 为 100 | `evaluation/robotwin.py` 默认 50 |
| 请求字段 | `json_numpy` 的 `proprio`、`language_instruction`、`domain_id`、三路图像 | 标准 JSON 的 `qpos` 和三路图像；domain id 来自 checkpoint |

因此本仓库 raw checkpoint 是关节位置策略。它和 `2toINF/X-VLA-RoboTwin2` 的末端位姿结果不能直接比较。

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
