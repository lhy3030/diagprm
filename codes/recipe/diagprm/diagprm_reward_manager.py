"""
DiagPRM - Multi-Turn Reward Manager

继承自 verl 的 AbstractRewardManager，重写 __call__ 以支持：
  1. Turn-level KG 覆盖率差分奖励
  2. 确诊奖励（episode 末）

奖励分配策略：
  - process_reward：放在该轮 end_position（密集信号）
  - outcome_reward：放在 final_turn end_position（稀疏信号）
  - adv_compute_info：传递给 compute_diagprm_turn_advantage 使用
"""

import json
import torch
import numpy as np
from collections import defaultdict
from typing import Dict, Any, Optional, Callable, List

from verl import DataProto
from verl.workers.reward_manager.abstract import AbstractRewardManager

from recipe.diagprm.diagprm_reward_fn import (
    parse_turns_from_response_mask,
    extract_human_responses,
    compute_episode_rewards,
    compute_episode_rewards_from_history,
)
from recipe.diagprm.kg_utils import load_kg


class DiagPRMRewardManager(AbstractRewardManager):
    """
    DiagPRM 的多轮 Reward Manager。

    初始化参数：
      tokenizer       : HF tokenizer
      num_examine     : 日志中展示的样本数
      kg_path         : master_kg.json 路径
      reward_params   : 各奖励系数字典（beta/r_max/tau 等）
    """

    def __init__(
        self,
        tokenizer,
        num_examine: int = 0,
        compute_score: Optional[Callable] = None,  # 兼容接口，DiagPRM 不使用
        reward_fn_key: str = "data_source",         # 兼容接口
        kg_path: str = "",
        reward_coefficients: Optional[Dict] = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.kg_path = kg_path

        # 奖励系数（可通过 config.reward_coefficients 覆盖）
        self.reward_params = {
            "beta": 1.0,        # KG 覆盖率差分系数
            "turn_coef": 1.0,   # Turn 奖励总系数：r(k) = turn_coef * r_turn + r_diag
            "r_max": 1.0,       # 正确诊断奖励
            "tau": 0.5,         # 过早确诊 KG 覆盖率阈值
            "weighted": True,
            "min_new_facts_for_diagnosis": 2,
            # Kept for backward-compatible configs; correct-but-premature now
            # receives r_max and is logged only as an analysis metric.
            "r_premature_diag": 0.0,
            "r_wrong_diag": 0.0,
            "r_timeout": -2.0,
        }
        if reward_coefficients:
            self.reward_params.update(reward_coefficients)

        # 延迟加载 KG（第一次调用 __call__ 时加载）
        self._kg = None

    def _get_kg(self):
        if self._kg is None:
            if not self.kg_path:
                raise ValueError(
                    "kg_path must be set in DiagPRMRewardManager. "
                    "Pass it via reward_model.reward_kwargs.kg_path in config."
                )
            self._kg = load_kg(self.kg_path)
        return self._kg

    def __call__(self, data: DataProto, return_dict: bool = False):
        """
        计算 batch 中每个样本的 turn-level reward。

        Returns:
            If return_dict=True:
                {"reward_tensor": outcome_reward_tensor, "reward_extra_info": {...}}
            Else:
                outcome_reward_tensor (shape: bs x response_length)
        """
        kg = self._get_kg()

        # 初始化奖励 tensor（shape: bs x response_length）
        process_reward_tensor = torch.zeros_like(
            data.batch["responses"], dtype=torch.float32
        )
        outcome_reward_tensor = torch.zeros_like(
            data.batch["responses"], dtype=torch.float32
        )
        reward_extra_info = defaultdict(list)

        for i in range(len(data)):
            data_item = data[i]

            response_ids = data_item.batch["responses"]
            response_mask = data_item.batch["response_mask"]

            # Get ground-truth disease name.
            # New schema: top-level "disease" field (string).
            # Old schema: reward_model.ground_truth.disease (dict).
            ground_truth = data_item.non_tensor_batch.get("disease", None)
            initial_symptoms = data_item.non_tensor_batch.get("initial_symptoms", None)
            gt_field = {}
            if not ground_truth:
                rm_info = data_item.non_tensor_batch.get("reward_model", {})
                if isinstance(rm_info, str):
                    try:
                        rm_info = json.loads(rm_info)
                    except Exception:
                        rm_info = {}
                gt_field = rm_info.get("ground_truth", "")
                if isinstance(gt_field, dict):
                    ground_truth = gt_field.get("disease", gt_field.get("answer", ""))
                    if initial_symptoms is None:
                        initial_symptoms = gt_field.get("initial_symptoms", [])
                else:
                    ground_truth = str(gt_field)
            ground_truth = str(ground_truth or "unknown")
            if isinstance(initial_symptoms, str):
                try:
                    initial_symptoms = json.loads(initial_symptoms)
                except Exception:
                    initial_symptoms = [initial_symptoms]
            if initial_symptoms is None:
                initial_symptoms = []

            num_turn = data_item.non_tensor_batch.get("__num_turns__", None)

            # ── 优先从 statistics.dialogue_history 读取对话轨迹（方案A）──────────
            # statistics 由 DiagPRMAgentLoop 写入，包含每轮的 doctor_response / patient_fact
            statistics = data_item.non_tensor_batch.get(
                "statistics",
                data_item.non_tensor_batch.get("diagprm_statistics", {}),
            )
            statistics = self._as_plain_dict(statistics)

            dialogue_history = statistics.get("dialogue_history", None)
            fact_id_to_text = statistics.get("fact_id_to_text", {}) if isinstance(statistics, dict) else {}
            if not isinstance(fact_id_to_text, dict):
                fact_id_to_text = {}

            if dialogue_history:
                # 方案A：直接从对话历史计算（推荐，无 token 解析歧义）
                total_rewards, r_diag_list, details_list = compute_episode_rewards_from_history(
                    dialogue_history=dialogue_history,
                    ground_truth=ground_truth,
                    kg=kg,
                    reward_params=self.reward_params,
                    initial_symptoms=initial_symptoms,
                    fact_id_to_text=fact_id_to_text,
                )
                # 位置仍从真实 full-history response_mask 解析，确保每轮 reward
                # 写回该轮 Doctor response 的 token span，而不是聚合到最后一轮。
                turns_info = parse_turns_from_response_mask(
                    response_mask, response_ids, self.tokenizer
                )
                if len(turns_info) != len(total_rewards):
                    print(
                        f"[Warning] Dialogue/token turn mismatch: "
                        f"history={len(total_rewards)}, mask={len(turns_info)}"
                    )
                    keep = min(len(turns_info), len(total_rewards))
                    turns_info = turns_info[:keep]
                    total_rewards = total_rewards[:keep]
                    r_diag_list = r_diag_list[:keep]
                    details_list = details_list[:keep]
            else:
                # 方案B fallback：从 response_mask 解析（传统多轮拼接模式）
                turns_info = parse_turns_from_response_mask(
                    response_mask, response_ids, self.tokenizer
                )
                human_responses = extract_human_responses(
                    response_ids, response_mask, self.tokenizer
                )
                if num_turn is not None and len(turns_info) != num_turn:
                    print(f"[Warning] Turn count mismatch: expected {num_turn}, got {len(turns_info)}")

                if not turns_info:
                    reward_extra_info["turn_count"].append(0)
                    reward_extra_info["process_rewards_sum"].append(0.0)
                    reward_extra_info["outcome_reward"].append(0.0)
                    reward_extra_info["adv_compute_info"].append({
                        "turn_end_positions": [],
                        "turn_process_rewards": [],
                        "turn_delta_kg_rewards": [],
                        "turn_actions": [],
                        "outcome_reward": 0.0,
                    })
                    reward_extra_info["diagprm_trajectory"].append(
                        self._build_diagprm_trajectory(
                            ground_truth=ground_truth,
                            initial_symptoms=initial_symptoms,
                            dialogue_history=[],
                            fact_id_to_text={},
                            details_list=[],
                            total_rewards=[],
                            final_r_diag=0.0,
                        )
                    )
                    continue

                total_rewards, r_diag_list, details_list = compute_episode_rewards(
                    turns_info=turns_info,
                    human_responses=human_responses,
                    ground_truth=ground_truth,
                    kg=kg,
                    reward_params=self.reward_params,
                    initial_symptoms=initial_symptoms,
                )

            if not turns_info:
                # turns_info 为空意味着 response_mask 中没有 mask=1 的 token span。
                # 若方案A已通过 dialogue_history 计算出了 rewards / details，
                # 则保留这些结果并把聚合 reward 写到最后一个有效 response 位置；
                # 否则（方案B fallback 且真正空序列）才全部清零。
                # dialogue_history 存在说明已走方案A，details_list / total_rewards 已被赋值
                _has_history_rewards = bool(dialogue_history)
                _final_r_diag = r_diag_list[-1] if (_has_history_rewards and r_diag_list) else 0.0
                _total_rew = total_rewards if _has_history_rewards else []
                _details = details_list if _has_history_rewards else []
                _aggregate_reward = sum(_total_rew) if _total_rew else 0.0

                # 兜底：无法定位逐轮 token span 时，只能把整条轨迹 reward 聚合到最后一个
                # response token。adv_compute_info 也保持单位置/单 reward，避免长度不一致。
                _mask_list = response_mask.tolist() if hasattr(response_mask, 'tolist') else list(response_mask)
                _last_pos = -1
                for _idx, _mask_val in enumerate(_mask_list):
                    if _mask_val == 1:
                        _last_pos = _idx
                if _has_history_rewards and _last_pos >= 0:
                    process_reward_tensor[i, _last_pos] = float(_aggregate_reward)
                    outcome_reward_tensor[i, _last_pos] = float(_final_r_diag)

                reward_extra_info["turn_count"].append(len(_details) if _details else 0)
                reward_extra_info["process_rewards_sum"].append(_aggregate_reward)
                reward_extra_info["outcome_reward"].append(_final_r_diag)
                reward_extra_info["adv_compute_info"].append({
                    "turn_end_positions": [_last_pos] if (_has_history_rewards and _last_pos >= 0) else [],
                    "turn_process_rewards": [_aggregate_reward] if (_has_history_rewards and _last_pos >= 0) else [],
                    "turn_delta_kg_rewards": [sum(float(d.get("delta_kg", 0.0)) for d in _details)] if (_has_history_rewards and _last_pos >= 0) else [],
                    "turn_actions": [str((_details[-1].get("action") if _details else "") or "")] if (_has_history_rewards and _last_pos >= 0) else [],
                    "outcome_reward": _final_r_diag,
                })
                reward_extra_info["diagprm_trajectory"].append(
                    self._build_diagprm_trajectory(
                        ground_truth=ground_truth,
                        initial_symptoms=initial_symptoms,
                        dialogue_history=dialogue_history or [],
                        fact_id_to_text=fact_id_to_text,
                        details_list=_details,
                        total_rewards=_total_rew,
                        final_r_diag=_final_r_diag,
                    )
                )
                continue

            # 将 total_reward 写入 tensor 的对应位置（end_position）
            # process_reward_tensor 存 turn_coef * r_turn（非最终轮的纯即时信号）
            # outcome_reward_tensor 存 r_diag（仅最终轮）
            turn_end_positions = []
            for turn_idx, turn_info in enumerate(turns_info):
                end_pos = turn_info["end_position"]
                turn_end_positions.append(end_pos)
                # total_reward = turn_coef * r_turn + r_diag，写入主 tensor
                process_reward_tensor[i, end_pos] = float(total_rewards[turn_idx])
                if turn_info["is_final_turn"]:
                    outcome_reward_tensor[i, end_pos] = float(r_diag_list[turn_idx])

            # 统计 metrics
            total_reward_sum = sum(total_rewards)
            final_r_diag = r_diag_list[-1] if r_diag_list else 0.0
            r_turn_list = [
                d.get("r_turn", 0.0) for d in details_list
            ]
            turn_delta_kg_rewards = [
                float(d.get("delta_kg", 0.0)) for d in details_list
            ]
            turn_actions = [
                str(d.get("action", "") or "").lower() for d in details_list
            ]
            reward_extra_info["turn_count"].append(len(turns_info))
            reward_extra_info["total_reward_sum"].append(total_reward_sum)
            reward_extra_info["total_reward_mean"].append(
                total_reward_sum / max(len(total_rewards), 1)
            )
            reward_extra_info["r_turn_sum"].append(sum(r_turn_list))
            reward_extra_info["r_diag"].append(final_r_diag)

            # 详细 metrics
            self._update_detailed_metrics(details_list, reward_extra_info)

            # adv_compute_info 供 Turn-level GRPO 使用。
            # 主方法不再按 hypothesis 分组；advantage 侧会按同一 prompt 的 turn index 分组。
            reward_extra_info["adv_compute_info"].append({
                "turn_end_positions": turn_end_positions,
                "turn_process_rewards": total_rewards,  # r(k) = turn_coef*r_turn + r_diag
                "turn_delta_kg_rewards": turn_delta_kg_rewards,  # raw Delta_KG for normalized turn advantage
                "turn_actions": turn_actions,  # A_turn is only applied to ask turns
                "outcome_reward": final_r_diag,
            })
            reward_extra_info["diagprm_trajectory"].append(
                self._build_diagprm_trajectory(
                    ground_truth=ground_truth,
                    initial_symptoms=initial_symptoms,
                    dialogue_history=dialogue_history or [],
                    fact_id_to_text=fact_id_to_text,
                    details_list=details_list,
                    total_rewards=total_rewards,
                    final_r_diag=final_r_diag,
                )
            )

        if return_dict:
            return {
                "reward_tensor": outcome_reward_tensor,
                "process_reward_tensor": process_reward_tensor,
                "reward_extra_info": dict(reward_extra_info),
            }
        else:
            return outcome_reward_tensor

    def _build_turns_info_from_history(
        self,
        dialogue_history: list,
        response_mask,
        response_ids,
    ) -> list:
        """
        状态摘要模式下，response_ids 只含最后一轮 Doctor token。
        构造一个单元素 turns_info，end_position 指向 response 末尾（mask=1 的最后位置）。
        这保证 reward tensor 写入到正确位置。
        """
        mask_list = response_mask.tolist() if hasattr(response_mask, 'tolist') else list(response_mask)
        # 找到最后一个 mask=1 的位置
        last_response_pos = -1
        for idx, m in enumerate(mask_list):
            if m == 1:
                last_response_pos = idx

        if last_response_pos == -1:
            return []

        # 虚拟 turns_info：每个 dialogue_history 条目对应一个虚拟轮次
        # 但 end_position 都指向同一个位置（状态摘要只有最后一轮 token）
        turns = []
        n = len(dialogue_history)
        for idx, entry in enumerate(dialogue_history):
            is_final = entry.get("is_final", idx == n - 1)
            turns.append({
                "turn_id": idx,
                "start_position": last_response_pos,
                "end_position": last_response_pos,
                "response": entry.get("doctor_response", ""),
                "is_final_turn": is_final,
                "length": 1,
            })
        return turns

    def _as_plain_dict(self, value) -> dict:
        """Convert numpy object wrappers / mapping-like values to a plain dict."""
        if isinstance(value, dict):
            return value
        if hasattr(value, "item"):
            try:
                item = value.item()
                if isinstance(item, dict):
                    return item
            except Exception:
                pass
        try:
            return dict(value) if value is not None else {}
        except Exception:
            return {}

    def _build_diagprm_trajectory(
        self,
        ground_truth: str,
        initial_symptoms: list,
        dialogue_history: list,
        fact_id_to_text: dict,
        details_list: list,
        total_rewards: list,
        final_r_diag: float,
    ) -> dict:
        """
        Compact JSON-serialisable trajectory for rollout inspection.

        `visible` is what the Doctor actually saw. `hidden` is oracle-side signal
        for reward/debug only and must not be fed back into the Doctor prompt.
        """
        turns = []
        for idx, entry in enumerate(dialogue_history or []):
            detail = details_list[idx] if idx < len(details_list) else {}
            fact_id = str(entry.get("patient_fact_id", "unknown") or "unknown")
            hidden_fact = fact_id_to_text.get(fact_id, entry.get("patient_fact_text", ""))
            turns.append({
                "turn_id": entry.get("turn_id", idx),
                "visible": {
                    "doctor_response": entry.get("doctor_response", ""),
                    "patient_answer": entry.get("patient_answer", ""),
                },
                "hidden": {
                    "patient_fact_id": fact_id,
                    "patient_fact_text": hidden_fact,
                    "is_final": bool(entry.get("is_final", idx == len(dialogue_history) - 1)),
                },
                "reward": {
                    "total": float(total_rewards[idx]) if idx < len(total_rewards) else 0.0,
                    "r_turn": float(detail.get("r_turn", 0.0)),
                    "delta_kg": float(detail.get("delta_kg", 0.0)),
                    "r_diag": float(detail.get("r_diag", 0.0)),
                    "coverage_after": float(detail.get("coverage_after", 0.0)),
                    "n_new_symptoms_collected": int(detail.get("n_new_symptoms_collected", 0)),
                },
            })
        return {
            "ground_truth": ground_truth,
            "initial_symptoms": initial_symptoms or [],
            "final_r_diag": float(final_r_diag),
            "turns": turns,
        }

    def _update_detailed_metrics(
        self,
        details_list: List[Dict],
        reward_extra_info: defaultdict,
    ) -> None:
        """统计每轮细节 metrics，写入 reward_extra_info。"""
        delta_kg_sum = 0.0
        n_valid_format = 0
        final_coverage = 0.0
        is_correct = False
        premature_diag = False

        for idx, d in enumerate(details_list):
            delta_kg_sum += d.get("delta_kg", 0.0)
            if d.get("has_valid_format"):
                n_valid_format += 1
            if d.get("is_correct_diagnosis"):
                is_correct = True
            if d.get("premature_diagnosis"):
                premature_diag = True
            final_coverage = d.get("coverage_after", final_coverage)

        n = max(len(details_list), 1)
        reward_extra_info["delta_kg_sum"].append(delta_kg_sum)
        reward_extra_info["valid_format_rate"].append(n_valid_format / n)
        reward_extra_info["final_kg_coverage"].append(final_coverage)
        reward_extra_info["is_correct"].append(1 if is_correct else 0)
        reward_extra_info["premature_diag_rate"].append(1 if premature_diag else 0)
