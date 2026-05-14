from __future__ import annotations
 
import argparse
import math
import os
from dataclasses import dataclass
 
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer, PreTrainedTokenizerBase

# MODEL

class RewardModel(nn.Module):
    """Decoder-only LLM backbone + scalar value head.
 
    The backbone is loaded via AutoModel (not AutoModelForCausalLM) — we don't
    need the vocabulary projection; we need the final hidden states. We then
    attach a single nn.Linear(H, 1, bias=False) head and read the scalar from
    the *last non-pad token's* final hidden state.
 
    Why the last non-pad token: in a causal/decoder-only transformer, attention
    is masked so each position i only attends to positions <= i. Only the last
    token's hidden state has attended to the entire sequence; all earlier
    positions have partial-context representations. For a sequence-level
    judgment ("how good is this response?"), the last position is the only
    correct readout.
 
    Equivalent in behavior to HuggingFace's
        AutoModelForSequenceClassification.from_pretrained(name, num_labels=1)
    but written out so the readout logic is explicit and inspectable.
    """
 
    def __init__(self, base_model_name: str, dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(base_model_name, torch_dtype=dtype)
        hidden_size = self.backbone.config.hidden_size
 
        # Single scalar head, no bias (Llama 2 / TRL convention).
        self.v_head = nn.Linear(hidden_size, 1, bias=False, dtype=dtype)
        # Small init keeps initial rewards near zero, avoiding large-magnitude
        # outputs at step 0 that would destabilize early gradients. This is
        # the initialization used in TRL and (effectively) Llama 2 RM training.
        nn.init.normal_(self.v_head.weight, std=1.0 / math.sqrt(hidden_size + 1))
 
    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.LongTensor,
    ) -> torch.Tensor:
        """
        Args:
            input_ids:      (B, T) token ids
            attention_mask: (B, T) 1 for real tokens, 0 for padding
        Returns:
            rewards: (B,) scalar reward per sequence
        """
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state  # (B, T, H)
 
        # Last non-pad token index per sequence. Assumes RIGHT padding.
        last_idx = attention_mask.sum(dim=1) - 1  # (B,)
        batch_idx = torch.arange(hidden.size(0), device=hidden.device)
        last_hidden = hidden[batch_idx, last_idx]  # (B, H)
 
        return self.v_head(last_hidden).squeeze(-1)  # (B,)
