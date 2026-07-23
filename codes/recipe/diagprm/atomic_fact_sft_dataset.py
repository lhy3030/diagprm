"""SFT dataset for ATPO raw prompt/response strings.

ATPO's SFT JSONL stores already-rendered chat-template strings in `prompt` and
`response`. The default verl SFTDataset would wrap `prompt` inside another user
message, so this dataset tokenizes the raw strings directly and masks prompt
tokens.
"""

from __future__ import annotations

from omegaconf.listconfig import ListConfig
import pandas as pd
import torch
from torch.utils.data import Dataset

from verl.utils.fs import copy_to_local
from verl.utils.model import compute_position_id_with_mask


class AtomicFactRawSFTDataset(Dataset):
    def __init__(self, parquet_files: str | ListConfig, tokenizer, config):
        if not isinstance(parquet_files, ListConfig):
            parquet_files = [parquet_files]
        self.parquet_files = list(parquet_files)
        self.tokenizer = tokenizer
        self.prompt_key = config.get("prompt_key", "prompt")
        self.response_key = config.get("response_key", "response")
        self.max_length = int(config.get("max_length", 4096))
        self.truncation = config.get("truncation", "error")
        self.use_shm = config.get("use_shm", False)
        if self.truncation not in {"error", "left", "right"}:
            raise ValueError(f"Unknown truncation mode: {self.truncation}")
        self._download()
        self._read()

    def _download(self) -> None:
        for i, parquet_file in enumerate(self.parquet_files):
            self.parquet_files[i] = copy_to_local(
                parquet_file, verbose=True, use_shm=self.use_shm
            )

    def _read(self) -> None:
        frames = [pd.read_parquet(path) for path in self.parquet_files]
        df = pd.concat(frames, ignore_index=True)
        self.prompts = df[self.prompt_key].astype(str).tolist()
        self.responses = df[self.response_key].astype(str).tolist()

    def __len__(self) -> int:
        return len(self.prompts)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        prompt = self.prompts[idx]
        response = self.responses[idx]
        if self.tokenizer.eos_token and self.tokenizer.eos_token not in response[-64:]:
            response = response + self.tokenizer.eos_token

        prompt_ids_out = self.tokenizer(
            prompt, return_tensors="pt", add_special_tokens=False
        )
        response_ids_out = self.tokenizer(
            response, return_tensors="pt", add_special_tokens=False
        )
        prompt_ids = prompt_ids_out["input_ids"][0]
        response_ids = response_ids_out["input_ids"][0]
        prompt_mask = prompt_ids_out["attention_mask"][0]
        response_mask = response_ids_out["attention_mask"][0]

        input_ids = torch.cat([prompt_ids, response_ids], dim=-1)
        attention_mask = torch.cat([prompt_mask, response_mask], dim=-1)
        prompt_len = int(prompt_ids.shape[0])
        response_len = int(response_ids.shape[0])

        if input_ids.shape[0] > self.max_length:
            if self.truncation == "error":
                raise NotImplementedError(
                    f"sequence_length={input_ids.shape[0]} exceeds max_length={self.max_length}"
                )
            if self.truncation == "left":
                input_ids = input_ids[-self.max_length :]
                attention_mask = attention_mask[-self.max_length :]
                # If left truncation cuts into prompt, all remaining response tokens
                # are still trainable after the shifted prompt boundary.
                prompt_len = max(0, prompt_len - (prompt_len + response_len - self.max_length))
            else:
                input_ids = input_ids[: self.max_length]
                attention_mask = attention_mask[: self.max_length]
        elif input_ids.shape[0] < self.max_length:
            pad_len = self.max_length - input_ids.shape[0]
            pad_id = self.tokenizer.pad_token_id
            input_ids = torch.cat([
                input_ids,
                torch.full((pad_len,), pad_id, dtype=input_ids.dtype),
            ])
            attention_mask = torch.cat([
                attention_mask,
                torch.zeros((pad_len,), dtype=attention_mask.dtype),
            ])

        loss_mask = attention_mask.clone()
        loss_mask[: min(prompt_len, loss_mask.shape[0])] = 0
        last_response_pos = min(prompt_len + response_len, loss_mask.shape[0]) - 1
        if last_response_pos >= 0:
            loss_mask[last_response_pos] = 0

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": compute_position_id_with_mask(attention_mask),
            "loss_mask": loss_mask,
        }
