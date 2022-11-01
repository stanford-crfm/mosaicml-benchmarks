# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""
Modified HF GPT2 using Flash Attention then wrapped for Composer
"""

import math
from typing import Any, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F
from composer.metrics.nlp import LanguageCrossEntropy, Perplexity
from composer.models.base import ComposerModel
from flash_attn.flash_attention import FlashMHA
from transformers.models.gpt2 import GPT2Config
from transformers.models.gpt2.modeling_gpt2 import GPT2LMHeadModel

from .hf_flash_gpt2 import GPT2FlashLMHeadModel


def prepare_hf_gpt2_model_for_fsdp(model):
    # Special Case! When using the LMHeadModel, the weights of the self.lm_head and self.transformer.wte are tied.
    # This tying occurs inside the `self.post_init()` function call above.
    # This is a hurdle for FSDP because they need to be in the same FSDP block
    # These lines ensures that both modules stay together in the top-most block
    model.transformer._fsdp_wrap = False
    model.transformer.wte._fsdp_wrap = False
    model.lm_head._fsdp_wrap = False

    # FSDP Wrap and Activation Checkpoint every GPT2Block
    for block in model.transformer.h:
        block._fsdp_wrap = True
        block._activation_checkpointing = True

class ComposerGPT(ComposerModel):

    def __init__(self, cfg):
        super().__init__()
        # load GPT2 config from standard HF model config json
        hf_config = GPT2Config.from_json_file(cfg.hf_config)
        # build model with config
        model_class = hf_config.architectures[0]
        if model_class == 'GPT2LMHeadModel':
            self.model = GPT2LMHeadModel(hf_config)
        elif model_class == 'GPT2FlashLMHeadModel':
            self.model = GPT2FlashLMHeadModel(hf_config)
        else:
            raise ValueError(f'Not sure how to build model_class={model_class}')

        # Tag layers to make the model ready for FSDP
        prepare_hf_gpt2_model_for_fsdp(self.model)

        self.train_metrics = {
            'LanguageCrossEntropy': LanguageCrossEntropy(hf_config.vocab_size),
            'Perplexity': Perplexity(),
        }
        self.eval_metrics = {
            'LanguageCrossEntropy': LanguageCrossEntropy(hf_config.vocab_size),
            'Perplexity': Perplexity(),
        }

    def get_targets(self, batch):
        targets = torch.roll(batch["labels"], shifts=-1)
        targets[:, -1] = -100
        return targets

    def forward(self, batch):
        return self.model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask']).logits

    def eval_forward(self, batch, outputs=None):
        return outputs if outputs is not None else self.forward(batch)

    def loss(self, outputs, batch):
        targets = self.get_targets(batch)
        return F.cross_entropy(outputs.view(-1, outputs.size(-1)),
                               targets.view(-1),
                               ignore_index=-100)

    def get_metrics(self, is_train=False):
        return self.train_metrics if is_train else self.eval_metrics

    def update_metric(self, batch, outputs, metric):
        outputs = outputs.view(-1, outputs.size(-1))
        targets = self.get_targets(batch).view(-1)
        metric.update(outputs, targets)
