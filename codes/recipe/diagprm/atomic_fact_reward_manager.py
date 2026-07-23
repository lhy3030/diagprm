"""Reward manager for ATPO-style atomic-fact medical QA RL."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from typing import Any, Callable, Optional

import torch

from verl import DataProto
from verl.workers.reward_manager.abstract import AbstractRewardManager
from recipe.diagprm.diagprm_reward_fn import parse_turns_from_response_mask


def _as_dict(value: Any) -> dict:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            obj = json.loads(value)
            return obj if isinstance(obj, dict) else {}
        except Exception:
            return {}
    if hasattr(value, "item"):
        try:
            obj = value.item()
            return obj if isinstance(obj, dict) else {}
        except Exception:
            return {}
    try:
        return dict(value) if value is not None else {}
    except Exception:
        return {}


def _strip_think(text: str) -> str:
    return re.sub(r"<think>[\s\S]*?</think>", "", text or "", flags=re.IGNORECASE).strip()


def parse_atomic_response(text: str) -> dict[str, Any]:
    stripped = _strip_think(text)
    q = re.search(r"^\s*Question\s*:\s*([\s\S]+?)\s*$", stripped, re.IGNORECASE)
    if q:
        return {"action": "ask", "question": q.group(1).strip()}
    ans = re.search(r"^\s*Final Answer\s*:\s*([A-Z])\b", stripped, re.IGNORECASE)
    if ans:
        return {"action": "answer", "answer": ans.group(1).upper()}
    return {"action": "invalid"}


class AtomicFactRewardManager(AbstractRewardManager):
    def __init__(
        self,
        tokenizer,
        num_examine: int = 0,
        compute_score: Optional[Callable] = None,
        reward_fn_key: str = "data_source",
        reward_coefficients: Optional[dict] = None,
        **kwargs,
    ) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.compute_score = compute_score
        self.reward_fn_key = reward_fn_key
        self.reward_params = {
            "beta": 1.0,
            "turn_coef": 0.4,
            "r_correct": 1.0,
            "r_wrong": 0.0,
            "r_timeout": -2.0,
        }
        if reward_coefficients:
            self.reward_params.update(dict(reward_coefficients))

    def __call__(self, data: DataProto, return_dict: bool = False):
        process_reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        outcome_reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)

        for i in range(len(data)):
            item = data[i]
            response_ids = item.batch["responses"]
            response_mask = item.batch["response_mask"]
            turns_info = parse_turns_from_response_mask(response_mask, response_ids, self.tokenizer)

            rm = _as_dict(item.non_tensor_batch.get("reward_model", {}))
            gt = _as_dict(rm.get("ground_truth", {}))
            answer_idx = str(
                item.non_tensor_batch.get("answer_idx")
                or gt.get("answer")
                or ""
            ).strip().upper()

            statistics = _as_dict(
                item.non_tensor_batch.get("statistics")
                or item.non_tensor_batch.get("diagprm_statistics", {})
            )
            dialogue_history = statistics.get("dialogue_history") or []
            fact_id_to_text = _as_dict(statistics.get("fact_id_to_text", {}))

            if dialogue_history:
                total_rewards, r_final_list, details = self._compute_from_history(
                    dialogue_history, fact_id_to_text, answer_idx
                )
                keep = min(len(turns_info), len(total_rewards))
                turns_info = turns_info[:keep]
                total_rewards = total_rewards[:keep]
                r_final_list = r_final_list[:keep]
                details = details[:keep]
            else:
                total_rewards, r_final_list, details = self._compute_from_tokens(
                    turns_info, answer_idx
                )

            if not turns_info:
                reward_extra_info["turn_count"].append(0)
                reward_extra_info["is_correct"].append(0)
                reward_extra_info["is_valid"].append(0)
                reward_extra_info["answer_rate"].append(0)
                reward_extra_info["fact_coverage"].append(0.0)
                reward_extra_info["delta_fact_sum"].append(0.0)
                reward_extra_info["adv_compute_info"].append({
                    "turn_end_positions": [],
                    "turn_process_rewards": [],
                    "turn_delta_kg_rewards": [],
                    "turn_actions": [],
                    "outcome_reward": 0.0,
                })
                continue

            turn_end_positions = []
            for turn_idx, turn in enumerate(turns_info):
                end_pos = turn["end_position"]
                turn_end_positions.append(end_pos)
                process_reward_tensor[i, end_pos] = float(total_rewards[turn_idx])
                if turn["is_final_turn"]:
                    outcome_reward_tensor[i, end_pos] = float(r_final_list[turn_idx])

            final_reward = r_final_list[-1] if r_final_list else 0.0
            turn_delta = [float(d.get("delta_fact", 0.0)) for d in details]
            turn_actions = [str(d.get("action", "")).lower() for d in details]
            delta_sum = sum(turn_delta)
            n = max(len(details), 1)
            is_correct = any(bool(d.get("is_correct")) for d in details)
            has_answer = any(d.get("action") == "answer" for d in details)
            valid_count = sum(1 for d in details if d.get("action") in {"ask", "answer"})
            ask_count = sum(1 for d in details if d.get("action") == "ask")
            no_eos_count = sum(1 for d in details if d.get("is_no_eos"))
            unknown_count = sum(1 for d in details if d.get("is_unknown_fact"))
            duplicate_count = sum(1 for d in details if d.get("is_duplicate_fact"))
            total_facts = max(int(details[-1].get("n_total_facts", 0)), 1)
            covered = int(details[-1].get("n_covered_facts", 0))

            reward_extra_info["turn_count"].append(len(details))
            reward_extra_info["total_reward_sum"].append(sum(total_rewards))
            reward_extra_info["total_reward_mean"].append(sum(total_rewards) / max(len(total_rewards), 1))
            reward_extra_info["r_turn_sum"].append(delta_sum)
            reward_extra_info["r_diag"].append(final_reward)
            reward_extra_info["is_correct"].append(1 if is_correct else 0)
            reward_extra_info["is_valid"].append(1 if has_answer else 0)
            reward_extra_info["answer_rate"].append(1 if has_answer else 0)
            reward_extra_info["valid_format_rate"].append(valid_count / n)
            reward_extra_info["no_eos_rate"].append(no_eos_count / n)
            reward_extra_info["fact_coverage"].append(covered / total_facts)
            reward_extra_info["delta_fact_sum"].append(delta_sum)
            reward_extra_info["unknown_fact_rate"].append(unknown_count / max(ask_count, 1))
            reward_extra_info["duplicate_fact_rate"].append(duplicate_count / max(ask_count, 1))
            reward_extra_info["adv_compute_info"].append({
                "turn_end_positions": turn_end_positions,
                "turn_process_rewards": total_rewards,
                # Reuse the DiagPRM advantage key name; here it means Delta_fact.
                "turn_delta_kg_rewards": turn_delta,
                "turn_actions": turn_actions,
                "outcome_reward": final_reward,
            })
            reward_extra_info["diagprm_trajectory"].append({
                "task": "atomic_fact_medqa",
                "answer_idx": answer_idx,
                "final_reward": float(final_reward),
                "fact_coverage": covered / total_facts,
                "turns": details,
            })

        if return_dict:
            return {
                "reward_tensor": outcome_reward_tensor,
                "process_reward_tensor": process_reward_tensor,
                "reward_extra_info": dict(reward_extra_info),
            }
        return outcome_reward_tensor

    def _compute_from_history(self, dialogue_history: list, fact_id_to_text: dict, answer_idx: str):
        fact_weights = {fid: 1.0 for fid in fact_id_to_text}
        total_weight = sum(fact_weights.values()) or 1.0
        covered: set[str] = set()
        total_rewards = []
        r_final_list = []
        details = []
        for idx, entry in enumerate(dialogue_history):
            parsed = parse_atomic_response(entry.get("doctor_response", ""))
            is_no_eos = bool(entry.get("is_no_eos") or entry.get("action") == "no_eos")
            action = "no_eos" if is_no_eos else parsed["action"]
            is_final = bool(entry.get("is_final", idx == len(dialogue_history) - 1))
            delta_fact = 0.0
            is_unknown = False
            is_duplicate = False
            if action == "ask":
                fid = str(entry.get("patient_fact_id", "unknown") or "unknown")
                if fid in fact_weights:
                    if fid in covered:
                        is_duplicate = True
                    else:
                        covered.add(fid)
                        delta_fact = self.reward_params["beta"] * fact_weights[fid] / total_weight
                else:
                    is_unknown = True
            r_final = 0.0
            is_correct = False
            if is_final:
                if action == "answer":
                    is_correct = str(parsed.get("answer", "")).upper() == answer_idx
                    r_final = self.reward_params["r_correct"] if is_correct else self.reward_params["r_wrong"]
                else:
                    r_final = self.reward_params["r_timeout"]
            total = self.reward_params["turn_coef"] * delta_fact + r_final
            details.append({
                "turn_id": entry.get("turn_id", idx),
                "action": action,
                "doctor_response": entry.get("doctor_response", ""),
                "patient_answer": entry.get("patient_answer", ""),
                "patient_fact_id": entry.get("patient_fact_id", "unknown"),
                "patient_fact_text": fact_id_to_text.get(str(entry.get("patient_fact_id", "")), ""),
                "delta_fact": float(delta_fact),
                "r_turn": float(delta_fact),
                "r_diag": float(r_final),
                "is_correct": bool(is_correct),
                "is_no_eos": bool(is_no_eos),
                "is_unknown_fact": bool(is_unknown and not is_final),
                "is_duplicate_fact": bool(is_duplicate and not is_final),
                "n_covered_facts": len(covered),
                "n_total_facts": len(fact_weights),
            })
            total_rewards.append(float(total))
            r_final_list.append(float(r_final))
        return total_rewards, r_final_list, details

    def _compute_from_tokens(self, turns_info: list, answer_idx: str):
        total_rewards = []
        r_final_list = []
        details = []
        for idx, turn in enumerate(turns_info):
            parsed = parse_atomic_response(turn.get("response", ""))
            action = parsed["action"]
            r_final = 0.0
            is_correct = False
            if turn.get("is_final_turn"):
                if action == "answer":
                    is_correct = str(parsed.get("answer", "")).upper() == answer_idx
                    r_final = self.reward_params["r_correct"] if is_correct else self.reward_params["r_wrong"]
                else:
                    r_final = self.reward_params["r_timeout"]
            details.append({
                "turn_id": idx,
                "action": action,
                "doctor_response": turn.get("response", ""),
                "delta_fact": 0.0,
                "r_turn": 0.0,
                "r_diag": float(r_final),
                "is_correct": bool(is_correct),
                "is_unknown_fact": False,
                "is_duplicate_fact": False,
                "n_covered_facts": 0,
                "n_total_facts": 1,
            })
            total_rewards.append(float(r_final))
            r_final_list.append(float(r_final))
        return total_rewards, r_final_list, details
