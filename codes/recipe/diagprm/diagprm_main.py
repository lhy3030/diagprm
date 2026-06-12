"""
DiagPRM - Training Entry Point

基于 ATPO 的 mt_main.py 修改：
  - 注册 DiagPRMRewardManager（替换 mt_reward_manager）
  - 注册 DiagPRMTrainer（支持 diagprm_grpo advantage）
  - 传入 kg_path 到 reward manager

运行方式：
  python -m recipe.diagprm.diagprm_main
  或通过 Hydra 配置：
  python -m recipe.diagprm.diagprm_main --config-name=diagprm_trainer
"""

import os
import socket
import multiprocessing
from functools import partial

import hydra
import ray
from omegaconf import OmegaConf

from verl.experimental.dataset.sampler import AbstractSampler
from verl.trainer.constants_ppo import get_ppo_ray_runtime_env
from verl.utils.device import is_cuda_available
from verl.utils.import_utils import load_extern_type
from verl.utils.reward_score import default_compute_score

from recipe.atpo.mt_main import (
    create_rl_dataset,
    create_rl_sampler,
    get_custom_reward_fn,
)
from recipe.atpo.mt_trainer import Role
from recipe.diagprm.diagprm_trainer import DiagPRMTrainer


def load_diagprm_reward_manager(config, tokenizer, num_examine, **reward_kwargs):
    """加载 DiagPRM Reward Manager。"""
    reward_manager_name = config.reward_model.get("reward_manager", "diagprm")

    if reward_manager_name == "diagprm":
        from recipe.diagprm.diagprm_reward_manager import DiagPRMRewardManager
        reward_manager_cls = DiagPRMRewardManager
    elif reward_manager_name == "mt_reward_manager":
        # 兼容 ATPO 的 reward manager
        from recipe.atpo.mt_reward_manager import MultiTurnRewardManager
        reward_manager_cls = MultiTurnRewardManager
    elif reward_manager_name == "naive":
        from verl.workers.reward_manager import NaiveRewardManager
        reward_manager_cls = NaiveRewardManager
    else:
        raise NotImplementedError(f"Unknown reward_manager: {reward_manager_name}")

    reward_coefficients = dict(config.get("reward_coefficients", {}))

    # kg_path 从 config 中读取
    kg_path = config.reward_model.get("kg_path", "")
    if not kg_path:
        # 尝试从默认位置读取
        default_kg_path = os.path.join(
            os.path.dirname(__file__),
            "../../../../origin_dataset/master_kg.json",
        )
        if os.path.exists(default_kg_path):
            kg_path = os.path.abspath(default_kg_path)
            print(f"[DiagPRM] Using default KG path: {kg_path}")
        else:
            print("[DiagPRM] WARNING: kg_path not set and default path not found!")

    return reward_manager_cls(
        tokenizer=tokenizer,
        num_examine=num_examine,
        kg_path=kg_path,
        reward_coefficients=reward_coefficients,
        **reward_kwargs,
    )


@hydra.main(config_path="config", config_name="diagprm_trainer", version_base=None)
def main(config):
    run_diagprm(config)


def run_diagprm(config) -> None:
    """初始化 Ray 并启动 DiagPRM 分布式训练。"""
    if not ray.is_initialized():
        default_runtime_env = get_ppo_ray_runtime_env()
        ray_init_kwargs = config.ray_kwargs.get("ray_init", {})
        runtime_env_kwargs = ray_init_kwargs.get("runtime_env", {})
        runtime_env = OmegaConf.merge(default_runtime_env, runtime_env_kwargs)
        ray_init_kwargs = OmegaConf.create({**ray_init_kwargs, "runtime_env": runtime_env})
        ray.init(**OmegaConf.to_container(ray_init_kwargs))

    runner = DiagPRMTaskRunner.remote()
    ray.get(runner.run.remote(config))


@ray.remote(num_cpus=1)
class DiagPRMTaskRunner:
    """Ray remote task runner for DiagPRM training."""

    def __init__(self):
        self.role_worker_mapping = {}
        self.mapping = {}

    def run(self, config):
        from pprint import pprint
        from omegaconf import OmegaConf
        from verl.utils.fs import copy_to_local
        from verl.utils import hf_processor, hf_tokenizer
        from verl.single_controller.ray import RayWorkerGroup
        from recipe.atpo.mt_trainer import ResourcePoolManager
        from verl.utils.dataset.rl_dataset import collate_fn

        print(f"DiagPRM TaskRunner: {socket.gethostname()}, PID: {os.getpid()}")
        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)

        local_path = copy_to_local(
            config.actor_rollout_ref.model.path,
            use_shm=config.actor_rollout_ref.model.get("use_shm", False),
        )
        # 新版 huggingface_hub 对 repo_id 格式有校验，不接受 /绝对路径 字符串。
        # 传入 pathlib.Path 对象可绕过该校验，transformers 会将其识别为本地路径。
        from pathlib import Path as _Path
        local_path_obj = _Path(local_path).resolve()

        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path_obj, trust_remote_code=trust_remote_code)
        processor = hf_processor(local_path_obj, trust_remote_code=trust_remote_code, use_fast=True)

        # Actor / Rollout Worker
        actor_rollout_cls, ray_worker_group_cls = self._add_actor_rollout_worker(config)
        self._add_ref_policy_worker(config, actor_rollout_cls)

        # Reward Manager
        reward_fn = load_diagprm_reward_manager(
            config, tokenizer, num_examine=0,
            **config.reward_model.get("reward_kwargs", {}),
        )
        val_reward_fn = load_diagprm_reward_manager(
            config, tokenizer, num_examine=1,
            **config.reward_model.get("reward_kwargs", {}),
        )

        # Resource Pool
        global_pool_id = "global_pool"
        resource_pool_spec = {
            global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        }
        self.mapping[Role.ActorRollout] = global_pool_id
        self.mapping[Role.Critic] = global_pool_id
        resource_pool_manager = ResourcePoolManager(
            resource_pool_spec=resource_pool_spec, mapping=self.mapping
        )

        # Datasets
        train_dataset = create_rl_dataset(config.data.train_files, config.data, tokenizer, processor, is_train=True)
        val_dataset = create_rl_dataset(config.data.val_files, config.data, tokenizer, processor, is_train=False)
        train_sampler = create_rl_sampler(config.data, train_dataset)

        # Trainer
        trainer = DiagPRMTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=self.role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            collate_fn=collate_fn,
            train_sampler=train_sampler,
        )
        trainer.init_workers()
        trainer.fit()

    def _add_actor_rollout_worker(self, config):
        from verl.single_controller.ray import RayWorkerGroup

        strategy = config.actor_rollout_ref.actor.strategy
        if strategy in {"fsdp", "fsdp2"}:
            if config.actor_rollout_ref.rollout.mode == "async":
                from verl.workers.fsdp_workers import AsyncActorRolloutRefWorker as cls
            else:
                from verl.workers.fsdp_workers import ActorRolloutRefWorker as cls
        elif strategy == "megatron":
            if config.actor_rollout_ref.rollout.mode == "async":
                from verl.workers.megatron_workers import AsyncActorRolloutRefWorker as cls
            else:
                from verl.workers.megatron_workers import ActorRolloutRefWorker as cls
        else:
            raise NotImplementedError

        self.role_worker_mapping[Role.ActorRollout] = ray.remote(cls)
        return cls, RayWorkerGroup

    def _add_ref_policy_worker(self, config, actor_rollout_cls):
        if config.algorithm.use_kl_in_reward or config.actor_rollout_ref.actor.use_kl_loss:
            self.role_worker_mapping[Role.RefPolicy] = ray.remote(actor_rollout_cls)
            self.mapping[Role.RefPolicy] = "global_pool"


if __name__ == "__main__":
    main()
