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

from .hf_flash_gpt import GPT2FlashLMHeadModel
from transformers.models.gpt2 import GPT2Config

class ComposerGPT(ComposerModel):

    def __init__(self, cfg, device='meta'):
        super().__init__()
        # load GPT2 config from standard HF model config json
        hf_config = GPT2Config.from_json_file(cfg.hf_config)
        # build model with config
        self.model = GPT2FlashLMHeadModel(hf_config)
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
        return self.model(input_ids=batch['input_ids']).logits

    def eval_forward(self, batch, outputs=None):
        return outputs if outputs is not None else self.forward(batch)

    def loss(self, outputs, batch):
        labels = batch['input_ids']
        shift_logits = lm_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        return F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), ignore_index=-100)
        #targets = self.get_targets(batch)
        #return F.cross_entropy(outputs.view(-1, outputs.size(-1)),
                               #targets.view(-1),
                               #ignore_index=-100)

    def get_metrics(self, is_train=False):
        return self.train_metrics if is_train else self.eval_metrics

    def update_metric(self, batch, outputs, metric):
        outputs = outputs.view(-1, outputs.size(-1))
        targets = self.get_targets(batch).view(-1)
        metric.update(outputs, targets)
