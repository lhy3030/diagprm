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
        advantages, returns = compute_diagprm_turn_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=data.batch["response_mask"],
            index=data.non_tensor_batch["uid"],
            norm_adv_by_std=norm_adv_by_std_in_grpo,
            adv_compute_info=data.non_tensor_batch.get("adv_compute_info", None),
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


class DiagPRMTrainer(RayPPOTrainer):
    """
    DiagPRM 的 Trainer。
    
    完全继承 ATPO RayPPOTrainer，仅重写 fit() 中 compute_advantage 的调用，
    注入 diagprm_grpo 支持。
    """

    def fit(self):
        """
        训练主循环，与 ATPO RayPPOTrainer.fit() 完全相同，
        唯一区别是 compute_advantage 调用使用本文件中的扩展版本。
        
        实现策略：通过猴子补丁将 mt_trainer 模块中的 compute_advantage
        临时替换为扩展版本，确保整个 fit() 流程中都使用新的实现。
        """
        import recipe.atpo.mt_trainer as mt_trainer_module
        _original = mt_trainer_module.compute_advantage
        mt_trainer_module.compute_advantage = compute_advantage
        try:
            super().fit()
        finally:
            mt_trainer_module.compute_advantage = _original
