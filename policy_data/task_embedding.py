import numpy as np
from omegaconf import OmegaConf

from configuration import load_paths
from tcla.dataloader.franka import _task_name_embedding, task_prompt_for
from tcla.utils.text_embedding import encode_texts_np


class TaskEmbEncoder:
    def __init__(self, config_path):
        cfg = OmegaConf.load(config_path)
        self.lang_dim = int(cfg.framework.lang_dim)
        dataset = OmegaConf.select(cfg, "datasets.action_data.dataset", default={})
        self.source = dataset.get("task_emb_source", "stored")
        self.cache = {}

    def encode(self, task_name):
        if task_name not in self.cache:
            if self.source == "stored":
                raise ValueError(
                    "This AE uses HDF5 task_emb; provide the training embedding at inference"
                )
            if self.source == "hash":
                embedding = _task_name_embedding(task_name, self.lang_dim)
            elif self.source == "minilm":
                embedding = encode_texts_np(
                    [task_prompt_for(task_name)],
                    name_or_path=load_paths().paths.language_model,
                    dim=self.lang_dim,
                    device="cpu",
                )[0]
            else:
                raise ValueError(f"Unsupported task embedding source: {self.source}")
            self.cache[task_name] = np.asarray(embedding, dtype=np.float32)
        return self.cache[task_name]
