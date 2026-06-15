"""
DiagPRM Trainer

在 ATPO 的 RayPPOTrainer 基础上，将优势计算替换为 Turn-level GRPO。
主要修改点：
  1. compute_advantage 中新增 "diagprm_grpo" 分支
  2. 调用 diagprm_algos.compute_diagprm_turn_advantage
  3. 其余流程与 ATPO 完全复用（rollout / actor update / checkpoint 等）
"""

from recipe.atpo.mt_trainer import (
    RayPPOTrainer,
    compute_advantage as _base_compute_advantage,
    AdvantageEstimator,
    Role,
    ResourcePoolManager,
)
from verl import DataProto
from verl.trainer.config import AlgoConfig
from typing import Optional

from recipe.diagprm.algo.diagprm_algos import compute_diagprm_turn_advantage
import numpy as np


# 注册新的 advantage estimator 名称
DIAGPRM_GRPO = "diagprm_grpo"


def compute_advantage(
    data: DataProto,
    adv_estimator,
    gamma: float = 1.0,
    lam: float = 1.0,
    num_repeat: int = 1,
    norm_adv_by_std_in_grpo: bool = True,
    config: Optional[AlgoConfig] = None,
) -> DataProto:
    """
    扩展 compute_advantage，新增 diagprm_grpo 分支。
    其余 estimator 类型代理给 ATPO 的原始实现。
    """
    if adv_estimator == DIAGPRM_GRPO or str(adv_estimator) == DIAGPRM_GRPO:
        # 从 config 中读取 GiGPO 混合系数 alpha（默认 0.5）
        alpha = 0.5
        if config is not None:
            alpha = getattr(config, "diagprm_alpha", 0.5)
        advantages, returns = compute_diagprm_turn_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=data.batch["response_mask"],
            index=data.non_tensor_batch["uid"],
            norm_adv_by_std=norm_adv_by_std_in_grpo,
            adv_compute_info=data.non_tensor_batch.get("adv_compute_info", None),
            alpha=alpha,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
        return data
    else:
        return _base_compute_advantage(
            data=data,
            adv_estimator=adv_estimator,
            gamma=gamma,
            lam=lam,
            num_repeat=num_repeat,
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
            config=config,
        )


def _diagprm_compute_accuracy_metrics(batch):
    """
    DiagPRM 版本的 compute_accuracy_metrics。
    
    UserAssistantAgentLoop 版本的 compute_accuracy_metrics 需要 is_valid / is_correct
    等字段，DiagPRM 的 reward manager 没有填充这些字段。
    这里用一个兼容版本替换，从 DiagPRM 自有的 reward_extra_info 字段中计算 metrics。
    """
    non_tensor = batch.non_tensor_batch
    n = len(batch)

    # 从 DiagPRM reward_extra_info 中读取已有的指标（如果存在）
    is_correct = non_tensor.get('is_correct', None)
    if is_correct is not None:
        num_correct = int(np.sum(is_correct))
    else:
        num_correct = 0

    is_valid_arr = non_tensor.get('is_valid', None)
    if is_valid_arr is not None:
        num_valid = int(np.sum(is_valid_arr))
    else:
        num_valid = n  # DiagPRM 默认所有样本有效

    valid_rate = num_valid / n if n > 0 else 0.0
    accuracy_of_valid = num_correct / num_valid if num_valid > 0 else 0.0
    is_correct_rate = num_correct / n if n > 0 else 0.0

    # 从 DiagPRM 特有的 metrics 中读取
    def _safe_mean(key):
        arr = non_tensor.get(key, None)
        if arr is not None and len(arr) > 0:
            return float(np.mean(arr))
        return 0.0

    return {
        "custom/num_sample": n,
        "custom/num_valid": num_valid,
        "custom/valid_rate": valid_rate,
        "custom/num_correct": num_correct,
        "custom/accuracy_of_valid": accuracy_of_valid,
        "custom/incorrect_format_rate": 1.0 - _safe_mean('valid_format_rate'),
        "custom/multiple_rate": 0.0,
        "custom/repeated_rate": 0.0,
        "custom/verify_timeout_rate": 0.0,
        "custom/effective_rate": valid_rate,
        "custom/human_reject_rate": 0.0,
        "custom/human_timeout_rate": 0.0,
        "custom/assistant_error_rate": 0.0,
        "custom/is_correct": is_correct_rate,
        # DiagPRM 特有 metrics
        "diagprm/total_reward_mean": _safe_mean('total_reward_mean'),
        "diagprm/r_diag_mean": _safe_mean('r_diag'),
        "diagprm/turn_count_mean": _safe_mean('turn_count'),
        "diagprm/final_kg_coverage_mean": _safe_mean('final_kg_coverage'),
        "diagprm/valid_format_rate": _safe_mean('valid_format_rate'),
        "diagprm/r_hyp_mean": _safe_mean('r_hyp_mean'),
        "diagprm/delta_kg_sum_mean": _safe_mean('delta_kg_sum'),
    }


class DiagPRMTrainer(RayPPOTrainer):
    """
    DiagPRM 的 Trainer。
    
    完全继承 ATPO RayPPOTrainer，仅重写 fit() 中 compute_advantage 的调用，
    注入 diagprm_grpo 支持。
    """

    def fit(self):
        """
        训练主循环，与 ATPO RayPPOTrainer.fit() 完全相同，
        唯一区别是 compute_advantage 调用使用本文件中的扩展版本，
        且 compute_accuracy_metrics 使用 DiagPRM 兼容版本。
        
        实现策略：通过猴子补丁将 mt_trainer 模块中的 compute_advantage
        临时替换为扩展版本，确保整个 fit() 流程中都使用新的实现。
        """
        import recipe.atpo.mt_trainer as mt_trainer_module
        _original_adv = mt_trainer_module.compute_advantage
        _original_acc = mt_trainer_module.compute_accuracy_metrics
        mt_trainer_module.compute_advantage = compute_advantage
        mt_trainer_module.compute_accuracy_metrics = _diagprm_compute_accuracy_metrics
        try:
            super().fit()
        finally:
            mt_trainer_module.compute_advantage = _original_adv
            mt_trainer_module.compute_accuracy_metrics = _original_acc
