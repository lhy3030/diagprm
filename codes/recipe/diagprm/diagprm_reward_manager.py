"""
DiagPRM - Multi-Turn Reward Manager

继承自 verl 的 AbstractRewardManager，重写 __call__ 以支持：
  1. Turn-level KG 覆盖率差分奖励
  2. 假设正确性奖励
  3. 假设切换修正奖励
  4. 确诊奖励（episode 末）

奖励分配策略：
  - process_reward：放在该轮 end_position（密集信号）
  - outcome_reward：放在 final_turn end_position（稀疏信号）
  - adv_compute_info：传递给 compute_grpo_turn_advantage 使用
"""

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
)
from recipe.diagprm.kg_utils import load_kg


class DiagPRMRewardManager(AbstractRewardManager):
    """
    DiagPRM 的多轮 Reward Manager。

    初始化参数：
      tokenizer       : HF tokenizer
      num_examine     : 日志中展示的样本数
      kg_path         : master_kg.json 路径
      reward_params   : 各奖励系数字典（beta/gamma1/lam/r_max/tau 等）
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
            "beta": 1.0,      # KG 覆盖率差分系数
            "gamma1": 0.3,    # 假设正确性系数
            "lam": 0.5,       # 切换修正系数
            "r_max": 2.0,     # 最大确诊奖励
            "tau": 0.5,       # 过早确诊 KG 覆盖率阈值
            "format_score": 0.1,
            "weighted": True,
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

            # 获取 ground truth 疾病名
            rm_info = data_item.non_tensor_batch.get("reward_model", {})
            ground_truth = rm_info.get("ground_truth", "")
            if isinstance(ground_truth, dict):
                ground_truth = ground_truth.get("disease", ground_truth.get("answer", ""))
            ground_truth = str(ground_truth)

            num_turn = data_item.non_tensor_batch.get("__num_turns__", None)

            # 解析 turn 边界
            turns_info = parse_turns_from_response_mask(
                response_mask, response_ids, self.tokenizer
            )
            # 解析患者回复
            human_responses = extract_human_responses(
                response_ids, response_mask, self.tokenizer
            )

            if num_turn is not None:
                if len(turns_info) != num_turn:
                    print(
                        f"[Warning] Turn count mismatch: expected {num_turn}, got {len(turns_info)}"
                    )

            if not turns_info:
                reward_extra_info["turn_count"].append(0)
                reward_extra_info["process_rewards_sum"].append(0.0)
                reward_extra_info["outcome_reward"].append(0.0)
                reward_extra_info["adv_compute_info"].append(
                    {"turn_end_positions": [], "turn_process_rewards": [], "outcome_reward": 0.0}
                )
                continue

            # 计算每轮 reward
            process_rewards, outcome_rewards, details_list = compute_episode_rewards(
                turns_info=turns_info,
                human_responses=human_responses,
                ground_truth=ground_truth,
                kg=kg,
                reward_params=self.reward_params,
            )

            # 将 reward 写入 tensor 的对应位置（end_position）
            turn_end_positions = []
            for turn_idx, turn_info in enumerate(turns_info):
                end_pos = turn_info["end_position"]
                turn_end_positions.append(end_pos)
                process_reward_tensor[i, end_pos] = float(process_rewards[turn_idx])
                if turn_info["is_final_turn"]:
                    outcome_reward_tensor[i, end_pos] = float(outcome_rewards[turn_idx])

            # 统计 metrics
            total_process = sum(process_rewards)
            final_outcome = outcome_rewards[-1] if outcome_rewards else 0.0
            reward_extra_info["turn_count"].append(len(turns_info))
            reward_extra_info["process_rewards_sum"].append(total_process)
            reward_extra_info["process_rewards_mean"].append(
                total_process / max(len(process_rewards), 1)
            )
            reward_extra_info["outcome_reward"].append(final_outcome)

            # 详细 metrics
            self._update_detailed_metrics(details_list, reward_extra_info)

            # adv_compute_info 供 Turn-level GRPO 使用
            reward_extra_info["adv_compute_info"].append({
                "turn_end_positions": turn_end_positions,
                "turn_process_rewards": process_rewards,
                "outcome_reward": final_outcome,
            })

        if return_dict:
            return {
                "reward_tensor": outcome_reward_tensor,
                "process_reward_tensor": process_reward_tensor,
                "reward_extra_info": dict(reward_extra_info),
            }
        else:
            return outcome_reward_tensor

    def _update_detailed_metrics(
        self,
        details_list: List[Dict],
        reward_extra_info: defaultdict,
    ) -> None:
        """统计每轮细节 metrics，写入 reward_extra_info。"""
        delta_kg_sum = 0.0
        r_hyp_sum = 0.0
        r_switch_count = 0
        correct_switches = 0
        wrong_switches = 0
        n_valid_format = 0
        final_coverage = 0.0
        is_correct = False

        for idx, d in enumerate(details_list):
            delta_kg_sum += d.get("delta_kg", 0.0)
            r_hyp_sum += d.get("r_hyp", 0.0)
            if d.get("action") == "switch":
                r_switch_count += 1
                if d.get("r_switch", 0.0) > 0:
                    correct_switches += 1
                elif d.get("r_switch", 0.0) < 0:
                    wrong_switches += 1
            if d.get("has_valid_format"):
                n_valid_format += 1
            if d.get("is_correct_diagnosis"):
                is_correct = True
            final_coverage = d.get("coverage_after", final_coverage)

        n = max(len(details_list), 1)
        reward_extra_info["delta_kg_sum"].append(delta_kg_sum)
        reward_extra_info["r_hyp_mean"].append(r_hyp_sum / n)
        reward_extra_info["switch_count"].append(r_switch_count)
        reward_extra_info["correct_switch_ratio"].append(
            correct_switches / max(r_switch_count, 1)
        )
        reward_extra_info["wrong_switch_ratio"].append(
            wrong_switches / max(r_switch_count, 1)
        )
        reward_extra_info["valid_format_rate"].append(n_valid_format / n)
        reward_extra_info["final_kg_coverage"].append(final_coverage)
        reward_extra_info["is_correct"].append(1 if is_correct else 0)
