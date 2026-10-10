# XVLA-TCLA

[中文](README_CN.md)

XVLA-TCLA integrates the TCLA action autoencoder with
[X-VLA](https://github.com/2toinf/X-VLA) for RoboTwin and Franka.
This project is an independent derivative of X-VLA. The vision-language
backbone and soft-prompted policy transformer come from X-VLA. TCLA adds action
compression, trajectory-conditioned decoding, and independent AE training.

## Workflows

| Mode | Training target | Autoencoder | Inference output |
| --- | --- | --- | --- |
| `raw` | Absolute joint actions and gripper | Not loaded; no AE checkpoint or statistics required | One physical action per request |
| `latent` | AE latent tokens, with an optional decoded-action loss | Pretrained and frozen | Latent tokens decoded to physical actions |

Raw RoboTwin uses 14 channels: left arm (6), left gripper, right arm (6), right
gripper. Franka uses 8 channels: seven joints and gripper. Raw joint losses retain
X-VLA's MSE for joints and BCE for grippers; training gripper targets must be in
`[0, 1]`. Latent mode uses the AE's action mean/std normalization.

## Installation

```bash
conda create -n xvla-tcla python=3.10 -y
conda activate xvla-tcla
pip install -r requirements.txt
```

Install a PyTorch/torchvision pair appropriate for your CUDA driver. Training
launchers require Bash. Run commands from this project directory. RoboTwin's
simulator dependencies and the Franka controller belong in their own environments.
No sibling TCLA checkout or external Python interpreter is required by the AE.

## Configure Paths

Edit [configs/paths.yaml](configs/paths.yaml), or create the ignored
`configs/paths.local.yaml` with only machine-specific overrides:

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

Relative filesystem paths are anchored to this project. Model settings accept
Hugging Face IDs or local directories; prefix a relative model directory with
`./`, for example `./checkpoints/X-VLA-Pt`. `XVLA_TCLA_PATHS_CONFIG` selects an
alternative private YAML. Output roots are also configured in `paths.output`.
Inspect resolved paths with `python configuration.py`.

## Data Format

Each task is a directory containing `episode_0.hdf5`, `episode_1.hdf5`, etc.

| HDF5 key | RoboTwin | Franka |
| --- | --- | --- |
| `action` | `[T, 14]`, absolute joints and grippers | `[T, 8]`, absolute joints and gripper |
| `observations/qpos` | `[T, 14]` | `[T, 8]` |
| `observations/images/<camera>` | RGB uint8 `[T, H, W, 3]` | RGB uint8 `[T, H, W, 3]` |
| Cameras, in order | `cam_high`, `cam_left_wrist`, `cam_right_wrist` | `front_cam`, `wrist_cam` |
| `task_emb` for AE/latent | Required, `[384]` or `[1, 384]` | Computed from task prompts using the AE configuration |

AE training does not read images. Raw training does not read `task_emb`.
The AE reserves `episode_0` for validation. Franka AE training requires
two complete chunks per episode (64 steps with the default chunk size).
For policy training, set `data.task_names=null` to discover all task directories,
including a 50-task RoboTwin dataset. AE tasks are set separately by `task_name`.
Keep task names and language embeddings consistent across AE and policy training.

## Train the Autoencoder

The local AE configs are [RoboTwin](configs/ae/robotwin.yaml) and
[Franka](configs/ae/franka.yaml), adapted from TCLA's autoencoder training configuration.
Defaults: 32 action steps, 48 DCT bins, 8 latent tokens of width 16, EMA target
encoder, language-conditioned Perceiver progress model, and constant learning rate.

```bash
bash scripts/train_ae.sh robotwin 0
bash scripts/train_ae.sh franka 0
bash scripts/train_ae.sh robotwin 0,1 num_freqs=24 trainer.max_train_steps=160000
```

Edit `framework` to configure the encoder/decoder dimensions and loss weights;
the retained component types are `tf_seq_cnn`, `tf_attn_w_nonScalarPhase`,
`direct_flow_w_nonScalarPhase`, and `perceiver`. Configure frequency bins through
`num_freqs` and chunk size through `n_action_steps`, so dataset and model agree.
Franka uses MiniLM by default. A hash-conditioned AE can be explicitly selected
with `datasets.action_data.dataset.task_emb_source=hash`; do not change the
embedding source when using an already trained checkpoint.

Each timestamped AE run saves `config.yaml`, `dataset_statistics.json`,
`checkpoints/steps_<step>_pytorch_model.pt` and validation CSVs.
Checkpoints include optimizer/scheduler state. To resume:

```bash
python train_ae.py --config_yaml configs/ae/robotwin.yaml \
  trainer.is_resume=true \
  trainer.resume_checkpoint=results/robotwin_ae/selected_run/checkpoints/steps_160000_pytorch_model.pt
```

After training, point `paths.action_ae.<platform>` to the actual run directory.
Policy training selects `action_ae.step` (default `160000`) and reads the saved
AE dimensions, frequency settings and normalization automatically. The integrated
policy uses absolute actions and requires latent width >= physical action width.

## Train the Policy

[base.yaml](configs/policy/base.yaml) holds shared optimizer and loss settings.
[robotwin.yaml](configs/policy/robotwin.yaml) and
[franka.yaml](configs/policy/franka.yaml) select data, tasks, cameras and domain ID.

```bash
bash scripts/train_policy.sh robotwin raw 0
bash scripts/train_policy.sh robotwin latent 0
bash scripts/train_policy.sh franka raw 0
bash scripts/train_policy.sh franka latent 0
bash scripts/train_policy.sh robotwin latent 0,1 action_ae.step=160000 training.batch_size=16
```

`training.batch_size` is per GPU. `MAIN_PROCESS_PORT` selects the distributed port;
`MIXED_PRECISION=bf16` enables policy mixed precision while AE operations stay fp32.
`training.freeze_steps` controls the initial VLA backbone learning-rate freeze,
not the AE. The AE is frozen for all of `latent` training.
`latent.decoded_loss_coef=0` disables the decoded-action training loss.

```bash
python train.py --config configs/policy/franka.yaml --mode latent --print_config
python train.py --config configs/policy/robotwin.yaml --mode raw training.iters=50000
```

Policy checkpoints are saved under `results/<platform>_policy/<mode>/ckpt-<step>`.
They contain model/processor files and `policy.json`. Latent checkpoints also
contain `action_ae_config.yaml`, `action_ae.pt`, `dataset_statistics.json`,
`qpos_stats.json` and `task_embeddings.json`. Move the entire checkpoint directory
for deployment. Policy checkpoints are inference exports, not optimizer resumes.

## Inference

```bash
python serve.py --checkpoint results/robotwin_policy/latent/ckpt-50000 --port 8000
python -m evaluation.robotwin --task adjust_bottle --episodes 50 --save_video
```

The server detects raw or latent mode from the checkpoint. Without
`--checkpoint`, it uses `paths.policy_checkpoint`. `--steps` sets VLA denoising
steps; `--execute_steps` controls how many actions are used before replanning
(default: the complete chunk). Launch the RoboTwin evaluator from this directory
using the simulator environment. It uses `paths.robotwin` and writes episode
results and success rates to `paths.eval_output`. `--task all` evaluates the tasks
listed in the checkpoint. The evaluator applies 14D joint commands for both modes.

For Franka, start `serve.py` with a Franka checkpoint and connect from your robot
controller using [FrankaPolicyClient](inference/franka.py):

```python
from inference.franka import FrankaPolicyClient

policy = FrankaPolicyClient("http://127.0.0.1:8000")
policy.reset("260830_bowl_stack")
action = policy.step(qpos, front_rgb, wrist_rgb)
```

`qpos` contains the current seven joint positions plus gripper; camera inputs are
RGB uint8 arrays. `action` is an absolute 8D target in the training data's units.
Connect it to the existing controller's command method; this repository does not
include a vendor-specific Franka hardware driver. Call `reset` at every episode
boundary, then execute one returned action per request. Latent history records
issued action targets. Use one server process per independently controlled robot.

The HTTP API exposes `GET /info`, `POST /reset`, and `POST /act`. Payload arrays
are standard JSON arrays. `/act` accepts `qpos` and `image0`, `image1`, etc. in the
checkpoint's camera order, and returns `{"action": [...]}`. Known task embeddings
are bundled when consistent across episodes; unseen tasks or episode-specific
RoboTwin embeddings require the matching `task_emb` in `/reset`.

## Raw Differences from Upstream X-VLA

This section compares RoboTwin `raw` training and inference with
[X-VLA](https://github.com/2toinf/X-VLA) RoboTwin2. The reference is the
fine-tuning command in that README, `train.py`, `RobotWin2Handler` in
`datasets/domain_handler/simulations.py`, and
`evaluation/robotwin-2.0/client.py`. `X-VLA-Pt` is configured with
`action_mode=ee6d` and `num_actions=30`.

These settings match the upstream README fine-tuning example: batch size 16 per
GPU, learning rate `1e-4`, `learning_coef=0.1`, 50000 iterations, 1000 freeze
steps, 2000 warmup steps, AdamW betas `(0.9, 0.95)`, zero weight decay, gradient
clip 1.0, and no cosine decay. The domain id is 6. The three cameras are head,
left wrist, and right wrist. Image augmentation is a 224 bicubic resize,
brightness/contrast/saturation jitter of 0.2, and ImageNet normalization.
Denoising uses 10 steps by default. Evaluation uses `demo_clean`, and seeds
start at `2000 * (seed + 1)`. The `train.py` argparse defaults are
`learning_coef=1.0` and `iters=1000000`; this repository follows the README
example.

### Training

| Item | Upstream X-VLA | This repository, raw |
| --- | --- | --- |
| Action | Absolute end-effector pose, 20D: xyz(3) + rot6d(6) + gripper(1) per arm | Absolute joints, 14D: 6 joints + gripper per arm |
| Action head | Continues the pretrained ee6d head | Reinitializes the joint head after the dimension and horizon change |
| Time window | `num_actions=30`. Samples up to 1 second at 30 Hz by interpolation; the current frame is proprioception and the next 30 steps are the action | `chunk_size=32`. Takes 32 consecutive rows from the current frame, including that frame's action, and repeats the last frame past the end |
| Gripper target | Converts the end-effector gripper from open=1 to closed=1 before BCE | Uses dataset joint grippers already in `[0, 1]`, with BCE on channels 6 and 13 |
| Tasks | RoboTwin2: 50 tasks, 50 demonstrations each | `configs/policy/robotwin.yaml`: 10 tasks, 50 episodes each |
| Precision | The launch command uses `--mixed_precision bf16` | `scripts/train_policy.sh` defaults to `MIXED_PRECISION=no` |
| Save interval | `train.py` defaults to 50000 steps | `configs/policy/base.yaml` uses 10000 steps |

Training reads the HDF5 `task_name`, for example `adjust bottle`. That matches
the upstream evaluator, which replaces underscores in the task name with spaces.

### Inference

| Item | Upstream X-VLA | This repository, raw |
| --- | --- | --- |
| Command | Converts the 20D end-effector action to position, quaternion, and gripper, then calls `action_type='ee'` | Sends a 14D joint target with `action_type='qpos'` |
| Language | `task_name.replace('_', ' ')`, for example `adjust bottle` | `evaluation/robotwin.py` passes only the directory name to `reset`, so the server uses `adjust_bottle` |
| Replanning | At most 10 policy queries, each executing the full 30-step end-effector chunk | The server caches the chunk and returns one step per `/act`; the default chunk is 32 and evaluation allows 3000 environment steps |
| Episodes per task | `eval_robotwin.sh` uses 100 | `evaluation/robotwin.py` defaults to 50 |
| Request fields | `json_numpy` `proprio`, `language_instruction`, `domain_id`, and three images | Standard JSON `qpos` and three images; the domain id comes from the checkpoint |

A raw checkpoint from this repository is a joint-position policy. Its results
are not directly comparable with `2toINF/X-VLA-RoboTwin2`.

## Attribution and Citation

Credit for the original X-VLA architecture and pretrained models belongs to
[2toINF and the X-VLA authors](https://github.com/2toinf/X-VLA). See the
[paper](https://arxiv.org/abs/2510.10274) and
[project page](https://thu-air-dream.github.io/X-VLA/). When using this derivative
in research, cite X-VLA and identify the TCLA integration separately:

```bibtex
@article{zheng2025x,
  title   = {X-VLA: Soft-Prompted Transformer as Scalable Cross-Embodiment Vision-Language-Action Model},
  author  = {Zheng, Jinliang and Li, Jianxiong and Wang, Zhihao and Liu, Dongxiu and Kang, Xirui
             and Feng, Yuchun and Zheng, Yinan and Zou, Jiayin and Chen, Yilun and Zeng, Jia and others},
  journal = {arXiv preprint arXiv:2510.10274},
  year    = {2025}
}
```

The original [Apache 2.0 license](LICENSE) is retained, together with
[NOTICE](NOTICE) and [third-party notices](THIRD_PARTY_NOTICES.md). Florence-2
copyright headers remain in its source files. Weights, datasets, simulators and
robot drivers have their own licenses. Private path overrides, data, weights and
generated outputs are excluded by `.gitignore`.
