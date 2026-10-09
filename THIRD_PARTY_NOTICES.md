# Third-Party Notices

## X-VLA

The vision-language-action backbone, processor, configuration, transformer,
action spaces, optimizer scheduling, and RoboTwin evaluation adapter derive
from [X-VLA](https://github.com/2toinf/X-VLA), Copyright 2025 2toINF.
The original [Apache License 2.0](LICENSE) and source notices are retained.
Modified upstream files are marked in their headers. XVLA-TCLA adds a local
TCLA action autoencoder, raw and latent policy workflows, portable checkpoints,
and shared path configuration. It is not an official X-VLA distribution.

The local input did not contain upstream Git metadata, so an exact upstream
commit cannot be identified from this source snapshot.

## Florence-2

`models/modeling_florence2.py` and `models/configuration_florence2.py` retain
Microsoft and Hugging Face's 2024 copyright notices and Apache 2.0 license
headers. They were supplied through the X-VLA source tree.

## TCLA and External Assets

`tcla/` contains the action autoencoder v3 components copied from the companion
TCLA project. ACT and Baku policy implementations are not included here.

Model weights, language encoders, RoboTwin, robot drivers, and datasets are
external assets. Their respective licenses and access conditions still apply;
the code license does not grant rights to redistribute those assets.

The X-VLA paper citation is provided in [README.md](README.md).
