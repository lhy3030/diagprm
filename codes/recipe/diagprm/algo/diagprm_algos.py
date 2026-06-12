"""
DiagPRM - Turn-level GRPO Advantage Estimator（GiGPO 风格两层 Grouping）

核心思想（来自 proposal §3.6）：

  奖励结构：
    r(k) = turn_coef * r_turn(k) + r_diag
    r_turn(k) = format_reward + Δ_k^kg + r_hyp(k)

  两层归一化（GiGPO 风格）：
    ① 假设感知 Group（Turn 级别）：
       - 同一 uid（prompt）下，所有 rollout 中处于"相同假设"下的各轮 r_turn(k)，
         归入同一 group，在组内做均值/方差归一化 → Â_turn(k)
       - 直觉：在当前假设语境下，哪些问诊行为更有效？
    ② 轨迹 Group（Episode 级别）：
       - 同一 uid（prompt）下，所有 rollout 的最终 r_diag，
         跨 rollout 做均值/方差归一化 → Â_diag
       - 直觉：哪条完整轨迹诊断结果更好？

  GiGPO 风格混合：
    Â(k) = Â_turn(k) + α * Â_diag
    - 每一轮的最终 advantage 是 turn 级信号 + 轨迹级信号的加权和
    - α 默认 0.5，可通过 alpha 参数调节

与标准 GRPO 的区别：
  - 标准 GRPO：整条轨迹所有 token 共享同一个 scalar advantage
  - DiagPRM GRPO：两层分组归一化后混合，turn 内 token 共享该 turn 的 advantage
  - 不需要 Critic，不需要折扣超参数

adv_compute_info 格式（由 DiagPRMRewardManager 生成）：
  {
    "turn_end_positions": List[int],
    "turn_process_rewards": List[float],   # r(k) = turn_coef*r_turn + r_diag（已合并）
    "outcome_reward": float,               # 最终轮 r_diag
    "turn_hypotheses": List[str],          # 可选：每轮的 primary hypothesis（规范化）
  }
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
    alpha: float = 0.5,                   # GiGPO 混合系数：Â = Â_turn + alpha * Â_diag
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    GiGPO 风格两层 Grouping 的 Turn-level GRPO Advantage 计算。

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
              "turn_process_rewards": List[float],  # 已含 turn_coef*r_turn + r_diag
              "outcome_reward": float,
              "turn_hypotheses": List[str],          # 可选，每轮 primary hypothesis
            }
            如果为 None，退化为标准 outcome-only GRPO。
        alpha: GiGPO 混合系数，控制轨迹级信号 Â_diag 的权重。
            Â(k) = Â_turn(k) + alpha * Â_diag
            - alpha=0.0 → 纯 turn 级信号
            - alpha=1.0 → 等权混合

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

        # ══════════════════════════════════════════════════════════════════════
        # Step 1：按 uid 分组，提取每条轨迹的 turn 信息
        # ══════════════════════════════════════════════════════════════════════
        # uid2samples: {uid: [(sample_idx, turn_end_positions, turn_rewards, outcome_reward, hypotheses)]}
        uid2samples = defaultdict(list)
        for sample_idx in range(bsz):
            uid = index[sample_idx]
            info = adv_compute_info[sample_idx]
            turn_end_positions = info.get("turn_end_positions", [])
            turn_process_rewards = info.get("turn_process_rewards", [])
            outcome_reward = info.get("outcome_reward", 0.0)
            # 每轮的 primary hypothesis（可选，用于假设感知分组）
            turn_hypotheses = info.get("turn_hypotheses", None)

            # 从 turn_process_rewards 中分离 r_turn 和 r_diag：
            # turn_process_rewards[k] = turn_coef * r_turn(k) + r_diag(k)
            # 对于非最终轮，r_diag(k) = 0，所以 r_turn_val = turn_process_rewards[k]
            # 对于最终轮，r_diag = outcome_reward，r_turn_val = turn_process_rewards[-1] - outcome_reward
            r_turn_values = list(turn_process_rewards)
            if r_turn_values:
                # 最后一轮去掉 r_diag，还原纯 r_turn 部分
                r_turn_values[-1] = r_turn_values[-1] - outcome_reward

            uid2samples[uid].append((
                sample_idx,
                turn_end_positions,
                r_turn_values,
                outcome_reward,
                turn_hypotheses,
            ))

        # ══════════════════════════════════════════════════════════════════════
        # Step 2：层① — 假设感知 Group（turn 级）归一化 → Â_turn
        # ══════════════════════════════════════════════════════════════════════
        # 同一 uid + 同一 hypothesis（或同一 turn 位置）的 r_turn 归入一个 group
        # 结果：adv_turn_map[(sample_idx, turn_idx)] = normalized_adv_turn

        adv_turn_map: dict = {}  # (sample_idx, turn_idx) → float

        for uid, samples in uid2samples.items():
            if len(samples) == 1:
                # 只有一个 rollout：turn advantage = 0（组内无法对比）
                sample_idx, turn_end_pos, r_turn_values, outcome_reward, hyps = samples[0]
                for t_idx in range(len(r_turn_values)):
                    adv_turn_map[(sample_idx, t_idx)] = 0.0
                    # 同样写入 advantage tensor，保证 token 区间有明确值（0.0）
                    if t_idx < len(turn_end_pos):
                        prev_end = turn_end_pos[t_idx - 1] if t_idx > 0 else -1
                        _fill_turn_advantage(
                            advantages, sample_idx, turn_end_pos[t_idx], 0.0,
                            response_mask, prev_turn_end=prev_end,
                        )
                continue

            # 收集每个 turn 位置的 r_turn，按 hypothesis 做分组 key
            # 分组 key：(uid, hypothesis_name) 如果有 hypothesis 信息，否则用 (uid, turn_idx)
            # 结构：group_key → [(sample_idx, turn_idx, r_turn_val)]
            hyp_group: dict = defaultdict(list)

            for sample_idx, turn_end_pos, r_turn_values, outcome_reward, hyps in samples:
                for t_idx, r_turn_val in enumerate(r_turn_values):
                    if hyps is not None and t_idx < len(hyps):
                        # 假设感知分组：同一 hypothesis 下的所有 turn 放一组
                        hyp_key = hyps[t_idx] if hyps[t_idx] else f"__unknown_{t_idx}"
                    else:
                        # 无 hypothesis 信息：退化为按 turn 位置分组
                        hyp_key = f"__turn_{t_idx}"
                    group_key = (uid, hyp_key)
                    hyp_group[group_key].append((sample_idx, t_idx, r_turn_val))

            # 对每个 group 独立做归一化
            for group_key, entries in hyp_group.items():
                if len(entries) < 2:
                    # 组内只有一个 turn：advantage = 0
                    for s_idx, t_idx, _ in entries:
                        adv_turn_map[(s_idx, t_idx)] = 0.0
                    continue

                r_vals = torch.tensor([e[2] for e in entries], dtype=torch.float32)
                mean_r = r_vals.mean()
                std_r = r_vals.std() if norm_adv_by_std else torch.tensor(1.0)

                for i_local, (s_idx, t_idx, r_val) in enumerate(entries):
                    if norm_adv_by_std:
                        adv_val = (r_val - mean_r.item()) / (std_r.item() + epsilon)
                    else:
                        adv_val = r_val - mean_r.item()
                    adv_turn_map[(s_idx, t_idx)] = float(adv_val)

        # ══════════════════════════════════════════════════════════════════════
        # Step 3：层② — 轨迹 Group（episode 级）归一化 → Â_diag
        # ══════════════════════════════════════════════════════════════════════
        # 同一 uid 下所有 rollout 的 outcome_reward（r_diag）跨 rollout 归一化
        # 结果：adv_diag_map[sample_idx] = normalized_adv_diag

        adv_diag_map: dict = {}  # sample_idx → float

        for uid, samples in uid2samples.items():
            outcome_list = [(s[0], s[3]) for s in samples]  # (sample_idx, outcome_reward)

            if len(outcome_list) == 1:
                adv_diag_map[outcome_list[0][0]] = 0.0
                continue

            r_diag_vals = torch.tensor([o[1] for o in outcome_list], dtype=torch.float32)
            mean_diag = r_diag_vals.mean()
            std_diag = r_diag_vals.std() if norm_adv_by_std else torch.tensor(1.0)

            for s_idx, r_diag_val in outcome_list:
                if norm_adv_by_std:
                    adv_val = (r_diag_val - mean_diag.item()) / (std_diag.item() + epsilon)
                else:
                    adv_val = r_diag_val - mean_diag.item()
                adv_diag_map[s_idx] = float(adv_val)

        # ══════════════════════════════════════════════════════════════════════
        # Step 4：GiGPO 混合 → Â(k) = Â_turn(k) + alpha * Â_diag
        #         并将 advantage 填充到对应的 token 区间
        # ══════════════════════════════════════════════════════════════════════
        for uid, samples in uid2samples.items():
            for sample_idx, turn_end_pos, r_turn_values, outcome_reward, hyps in samples:
                adv_diag = adv_diag_map.get(sample_idx, 0.0)

                for t_idx, end_pos in enumerate(turn_end_pos):
                    adv_turn = adv_turn_map.get((sample_idx, t_idx), 0.0)
                    # GiGPO 混合公式
                    final_adv = adv_turn + alpha * adv_diag
                    # 通过 turn_end_pos 列表精确推算本 turn 的起始位置：
                    # 本 turn start = 前一个 turn end + 1（跳过中间的 human token 段由 response_mask 决定）
                    prev_end = turn_end_pos[t_idx - 1] if t_idx > 0 else -1
                    _fill_turn_advantage(
                        advantages, sample_idx, end_pos, final_adv, response_mask,
                        prev_turn_end=prev_end,
                    )

    return advantages, advantages  # returns = advantages（无折扣）


def _fill_turn_advantage(
    advantages: torch.Tensor,
    sample_idx: int,
    end_pos: int,
    adv_val: float,
    response_mask: torch.Tensor,
    prev_turn_end: int = -1,
) -> None:
    """
    将该 turn 的 advantage 值填充到 [turn_start, end_pos] 的所有 response token。

    策略：
    - 从 end_pos 向前扫，找到第一个 response_mask=1 的连续段的起始位置
    - 但不越过 prev_turn_end（上一个 turn 的结束位置），避免覆盖前一 turn 的 advantage

    Args:
        prev_turn_end: 前一个 turn 的 end_position（-1 表示本 turn 是第一个）。
                       起始扫描不越过此位置，防止跨 turn 覆盖。
    """
    seq_len = advantages.shape[1]
    if end_pos < 0 or end_pos >= seq_len:
        return

    # 向前扫描：找到该 turn 的起始位置
    # 停止条件：遇到 response_mask=0（human token）或到达 prev_turn_end 的边界
    start_pos = end_pos
    while start_pos > 0:
        prev = start_pos - 1
        # 不越过前一个 turn 的结束位置
        if prev <= prev_turn_end:
            break
        # 遇到 human token（mask=0）则停止
        if response_mask[sample_idx, prev].item() != 1:
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
