"""
DiagPRM - Turn-level GRPO Advantage Estimator

核心思想（来自 proposal §3.6）：
  每一轮独立作为一个 GRPO "回答"，用该轮的即时奖励直接加权。
  
  A_hat(k, i) = (r(k, i) - mean_j[r(k, j)]) / std_j[r(k, j)]
  
  其中 j 遍历同一历史节点（同一 prompt uid）的所有 G 个 rollout。

与标准 GRPO 的区别：
  - 标准 GRPO：整条轨迹所有 token 共享同一个 scalar advantage
  - DiagPRM GRPO：每个 turn 的 token 用该 turn 自己的即时奖励独立加权
  - 不需要 Critic，不需要折扣超参数

实现策略：
  adv_compute_info 由 DiagPRMRewardManager 生成，包含：
    {
      "turn_end_positions": [pos_turn0, pos_turn1, ...],
      "turn_process_rewards": [r0, r1, ...],
      "outcome_reward": float   # final turn 的确诊奖励
    }
  
  本函数将 process_reward + outcome_reward 写入对应的 token 位置，
  然后在 uid 组内对每个 turn position 独立归一化。
"""

from collections import defaultdict
from typing import Optional

import numpy as np
import torch


def compute_diagprm_turn_advantage(
    token_level_rewards: torch.Tensor,   # (bs, response_length)
    response_mask: torch.Tensor,          # (bs, response_length)
    index: np.ndarray,                    # (bs,) group index，相同 uid 的样本在同一组
    epsilon: float = 1e-6,
    norm_adv_by_std: bool = True,
    adv_compute_info: Optional[list] = None,  # List[dict]，由 reward manager 生成
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Turn-level GRPO Advantage 计算。

    Args:
        token_level_rewards: 已由 reward manager 在 end_position 处写入 reward 的 tensor。
            shape: (bs, response_length)
            格式：只有每轮 end_position 处有非零值，其余为 0。
        response_mask: 1 = 模型 token，0 = padding/human token。
            shape: (bs, response_length)
        index: 组 index，同一个 prompt（uid）的 G 个 rollout 有相同 index。
        epsilon: 防止除 0。
        norm_adv_by_std: 是否按组内 std 归一化（与 GRPO 一致）。
        adv_compute_info: List[dict]，每个样本对应一个 dict：
            {
              "turn_end_positions": List[int],
              "turn_process_rewards": List[float],
              "outcome_reward": float,
            }
            如果为 None，退化为标准 outcome-only GRPO。

    Returns:
        advantages: shape (bs, response_length)
            每个模型 token 位置的 advantage 值（turn 内所有 token 共享该 turn 的 advantage）
        returns: shape (bs, response_length)
            同 advantages（DiagPRM 中 return = advantage，无折扣）
    """
    bsz, seq_len = token_level_rewards.shape
    advantages = torch.zeros_like(token_level_rewards)

    with torch.no_grad():
        if adv_compute_info is None:
            # 退化到标准 GRPO（按整条轨迹 outcome 计算）
            return _fallback_grpo(
                token_level_rewards, response_mask, index, epsilon, norm_adv_by_std
            )

        # ── Step 1：收集每条轨迹的每个 turn 的 reward ─────────────────────────
        # 结构：{uid_index: {turn_pos: [reward_list_across_group]}}
        # turn_pos 用 end_position 标识，确保组内对应轮次可对齐

        # 先按 uid（index）分组，收集每个样本的 turn 信息
        uid2samples = defaultdict(list)  # {uid: [(sample_idx, turn_end_positions, turn_rewards)]}
        for sample_idx in range(bsz):
            uid = index[sample_idx]
            info = adv_compute_info[sample_idx]
            turn_end_positions = info.get("turn_end_positions", [])
            turn_process_rewards = info.get("turn_process_rewards", [])
            outcome_reward = info.get("outcome_reward", 0.0)

            # 把 process_reward + outcome_reward 合并：最后一轮的 reward = process + outcome
            combined_rewards = list(turn_process_rewards)
            if combined_rewards:
                combined_rewards[-1] = combined_rewards[-1] + outcome_reward

            uid2samples[uid].append((sample_idx, turn_end_positions, combined_rewards))

        # ── Step 2：对每个 uid 组内，按 turn 位置独立归一化 ────────────────────
        for uid, samples in uid2samples.items():
            if len(samples) == 1:
                # 只有一个样本：advantage = 0（无法组内对比）
                sample_idx, turn_end_pos, rewards = samples[0]
                for t_idx, (end_pos, r) in enumerate(zip(turn_end_pos, rewards)):
                    # advantage = 0，但仍需填入 response_mask 对应范围
                    _fill_turn_advantage(advantages, sample_idx, end_pos, 0.0, response_mask)
                continue

            # 找出组内所有样本的 turn 数量
            # 使用 min_turns 确保可以对齐（不同样本 turn 数可能不同）
            n_turns_list = [len(s[1]) for s in samples]
            # 对 turn 数目不匹配的情况：按最小 turn 数对齐（简化处理）
            # TODO：更精确的方案是按轨迹节点（Tree Rollout）对齐

            max_turns = max(n_turns_list)
            # 按 turn index 收集组内奖励
            for t_idx in range(max_turns):
                turn_rewards_in_group = []
                valid_samples = []
                for sample_idx, turn_end_pos, combined_rewards in samples:
                    if t_idx < len(combined_rewards):
                        turn_rewards_in_group.append(combined_rewards[t_idx])
                        valid_samples.append((sample_idx, turn_end_pos[t_idx] if t_idx < len(turn_end_pos) else None))

                if len(turn_rewards_in_group) < 2:
                    # 组内只有一个样本在这个 turn：advantage = 0
                    for s_idx, end_pos in valid_samples:
                        if end_pos is not None:
                            _fill_turn_advantage(advantages, s_idx, end_pos, 0.0, response_mask)
                    continue

                # 计算组内均值和标准差
                rewards_tensor = torch.tensor(turn_rewards_in_group, dtype=torch.float32)
                mean = rewards_tensor.mean()
                std = rewards_tensor.std() if norm_adv_by_std else torch.tensor(1.0)

                for i_local, (s_idx, end_pos) in enumerate(valid_samples):
                    if end_pos is None:
                        continue
                    r = turn_rewards_in_group[i_local]
                    if norm_adv_by_std:
                        adv_val = (r - mean.item()) / (std.item() + epsilon)
                    else:
                        adv_val = r - mean.item()
                    _fill_turn_advantage(advantages, s_idx, end_pos, float(adv_val), response_mask)

    return advantages, advantages  # returns = advantages（无折扣）


def _fill_turn_advantage(
    advantages: torch.Tensor,
    sample_idx: int,
    end_pos: int,
    adv_val: float,
    response_mask: torch.Tensor,
) -> None:
    """
    将该 turn 的 advantage 值填充到 [turn_start, end_pos] 的所有 response token。
    
    策略：向前扫描直到遇到 response_mask=0 或另一个 turn 的 end_pos。
    简化实现：从 end_pos 向前扫描，将连续的 response_mask=1 段都赋值。
    """
    seq_len = advantages.shape[1]
    if end_pos < 0 or end_pos >= seq_len:
        return

    # 找到该 turn 的起始位置（向前扫 response_mask=1 的连续段）
    start_pos = end_pos
    while start_pos > 0 and response_mask[sample_idx, start_pos - 1].item() == 1:
        # 如果上一个位置已经有非零 advantage（属于别的 turn），停止
        if advantages[sample_idx, start_pos - 1].item() != 0.0:
            break
        start_pos -= 1

    advantages[sample_idx, start_pos:end_pos + 1] = adv_val


def _fallback_grpo(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float,
    norm_adv_by_std: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """退化到标准 outcome GRPO：整条轨迹共享一个 scalar advantage。"""
    scores = token_level_rewards.sum(dim=-1)  # (bs,)
    id2scores = defaultdict(list)

    bsz = scores.shape[0]
    for i in range(bsz):
        id2scores[index[i]].append(scores[i])

    id2mean = {}
    id2std = {}
    for uid, sc_list in id2scores.items():
        if len(sc_list) == 1:
            id2mean[uid] = torch.tensor(0.0)
            id2std[uid] = torch.tensor(1.0)
        else:
            t = torch.stack(sc_list)
            id2mean[uid] = t.mean()
            id2std[uid] = t.std()

    normalized = scores.clone()
    for i in range(bsz):
        uid = index[i]
        if norm_adv_by_std:
            normalized[i] = (scores[i] - id2mean[uid]) / (id2std[uid] + epsilon)
        else:
            normalized[i] = scores[i] - id2mean[uid]

    advantages = normalized.unsqueeze(-1) * response_mask
    return advantages, advantages
