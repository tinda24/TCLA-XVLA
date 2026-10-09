# ------------------------------------------------------------------------------
# Copyright 2025 2toINF (https://github.com/2toINF)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Modified for XVLA-TCLA: scoped platform support and TCLA integration.
# ------------------------------------------------------------------------------

from transformers import ProcessorMixin
from typing import List, Union, Dict, Any, Optional
import torch


class XVLAProcessor(ProcessorMixin):
    num_views: int = 3
    language_max_length: int = 50

    attributes = ["image_processor", "tokenizer"]
    image_processor_class = "AutoImageProcessor"
    tokenizer_class = ("BartTokenizer", "BartTokenizerFast")

    def __init__(self, image_processor=None, tokenizer=None):

        super().__init__(image_processor, tokenizer)

    def encode_language(
        self, language_instruction: Union[str, List[str]]
    ) -> Dict[str, torch.Tensor]:

        if isinstance(language_instruction, str):
            language_instruction = [language_instruction]

        inputs = self.tokenizer(
            language_instruction,
            return_tensors="pt",
            padding="max_length",
            max_length=self.language_max_length,
            truncation=True,
        )
        return {"input_ids": inputs["input_ids"]}

    def encode_image(
        self, images: Union[List, List[List]], **kwargs
    ) -> Dict[str, torch.Tensor]:

        if not isinstance(images[0], (list, tuple)):
            images = [images]

        batch_imgs, batch_masks = [], []

        for sample_imgs in images:
            processed = self.image_processor(
                sample_imgs, return_tensors="pt", **kwargs
            )["pixel_values"]
            V_exist = processed.size(0)

            if V_exist < self.num_views:
                processed = torch.cat(
                    [
                        processed,
                        processed.new_zeros(
                            self.num_views - V_exist, *processed.shape[1:]
                        ),
                    ],
                    dim=0,
                )

            image_mask = torch.zeros(
                self.num_views, dtype=torch.bool, device=processed.device
            )
            image_mask[:V_exist] = True

            batch_imgs.append(processed)
            batch_masks.append(image_mask)

        image_input = torch.stack(batch_imgs, dim=0)
        image_mask = torch.stack(batch_masks, dim=0)

        return {"image_input": image_input, "image_mask": image_mask}

    def __call__(
        self,
        images: Optional[Union[List, List[List]]] = None,
        language_instruction: Optional[Union[str, List[str]]] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:

        outputs: Dict[str, Any] = {}

        if language_instruction is not None:
            outputs.update(self.encode_language(language_instruction))

        if images is not None:
            outputs.update(self.encode_image(images, **kwargs))

        if "input_ids" in outputs and "image_input" in outputs:
            assert outputs["input_ids"].size(0) == outputs["image_input"].size(0), (
                f"Batch mismatch: text batch {outputs['input_ids'].size(0)} "
                f"!= image batch {outputs['image_input'].size(0)}"
            )
        return outputs
