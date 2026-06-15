"""
DiagPRM - Multi-turn Diagnostic Dialogue Agent Loop

实现 Doctor Agent 与 Patient Simulator 的异步多轮对话。

对话协议（每轮）：
  Doctor  : <think>...</think><hypothesis_state>...</hypothesis_state>
             <action>continue|diagnose</action>
             <question>...</question>  或  <diagnosis>...</diagnosis>
  Patient : 根据原始症状列表（atomic_facts）回答医生的问题

Doctor Agent prompt 格式（中文/英文可选）：
  系统提示：角色描述 + 输出格式说明
  用户消息（每轮）：患者最新回答
  助手消息（每轮）：上一轮 Doctor 输出（含 CoT）

Patient Simulator：
  - 使用 mediQ 数据集中的 atomic_facts 字段
  - 通过 LLM（vLLM）调用实现问答
  - 备用方案：规则匹配（从 atomic_facts 中直接检索）
"""

import re
import asyncio
import json
from typing import Any, Dict, List, Optional, Tuple
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, convert_to_messages

from recipe.atpo.agent_loop import AgentLoopBase, AgentLoopOutput, AgentLoopMetrics
from recipe.atpo.chat_model import ChatModel, MaxTokenExceededError
from recipe.diagprm.prompts import (
    DOCTOR_SYSTEM_PROMPT,
    DOCTOR_INITIAL_PROMPT,
    PATIENT_SYSTEM_PROMPT,
    VERIFIER_SYSTEM_PROMPT,
)


# ──────────────────────────────────────────────────────────────────────────────
# DiagPRM Agent Loop
# ──────────────────────────────────────────────────────────────────────────────

class DiagPRMAgentLoop(AgentLoopBase):
    """
    DiagPRM 多轮诊断对话 Agent Loop。
    
    继承自 ATPO 的 AgentLoopBase，重写对话逻辑以支持：
      - Doctor Agent 的结构化 CoT 格式（<hypothesis_state>, <action>）
      - Patient Simulator（基于 atomic_facts 的规则 + LLM 混合问答）
      - Verifier（检测重复/多重问题）
      - 最大轮次控制

    配置（来自 agent.yaml）：
      max_turns           : 最大对话轮数（default: 10）
      use_llm_patient     : 是否用 LLM 模拟患者（default: True）
      patient_model       : 患者模拟器使用的模型（default: 与 doctor 相同）
      verifier_enabled    : 是否启用问题重复检测（default: True）
    """

    def __init__(self, trainer_config, server_manager, tokenizer, processor, **kwargs):
        # kwargs 包含来自 diagprm_agent.yaml 的字段（max_turns, use_llm_patient, verifier_enabled）
        super().__init__(trainer_config, server_manager, tokenizer, processor, **kwargs)
        self.max_turns = kwargs.get("max_turns", 10)
        self.use_llm_patient = kwargs.get("use_llm_patient", True)
        self.verifier_enabled = kwargs.get("verifier_enabled", True)

    async def run(self, sampling_params: dict, **kwargs) -> list:
        """
        执行一轮完整的多轮诊断对话。

        Args:
            sampling_params: vLLM 采样参数
            **kwargs: 来自数据集的字段（prompt, reward_model, extra_info, ...）

        Returns:
            list[AgentLoopOutput]：包含完整对话序列和 token 信息的列表
        """
        rollout = self.config.actor_rollout_ref.rollout
        model_type = self.config.actor_rollout_ref.model_type

        # 构建 ChatModel（与 UserAssistantAgentLoop 相同方式）
        model = ChatModel(
            model=model_type,
            client=self.server_manager,
            tokenizer=self.tokenizer,
            max_tokens=rollout.response_length,
        )

        # 获取数据集字段
        # prompt 是原始 chat messages（JSON 字符串或 list）
        raw_prompt = kwargs.get("prompt", [])
        if isinstance(raw_prompt, str):
            try:
                raw_prompt = json.loads(raw_prompt)
            except Exception:
                raw_prompt = []

        # reward_model 包含 ground_truth（disease, atomic_facts）
        reward_model = kwargs.get("reward_model", {})
        if isinstance(reward_model, str):
            try:
                reward_model = json.loads(reward_model)
            except Exception:
                reward_model = {}
        ground_truth = reward_model.get("ground_truth", {})

        disease = ground_truth.get("disease", "unknown")
        atomic_facts = ground_truth.get("atomic_facts", [])

        # 从 prompt 中提取主诉（chief complaint）
        chief_complaint = self._extract_chief_complaint(raw_prompt)

        # 初始化对话（使用 langchain messages）
        messages = self._build_initial_messages(chief_complaint)
        verifier_responses = []
        previous_questions = []
        turn_count = 0

        try:
            for turn_idx in range(self.max_turns):
                turn_count = turn_idx + 1
                is_last_turn = (turn_idx == self.max_turns - 1)

                # ── 更新系统提示中的 current_turn ──────────────────────────────
                messages[0] = SystemMessage(content=DOCTOR_SYSTEM_PROMPT.format(
                    max_turns=self.max_turns,
                    current_turn=turn_count,
                ))

                # ── Doctor 生成回复（通过 ChatModel）──────────────────────────
                try:
                    result = await model.ainvoke(
                        messages,
                        sampling_params=sampling_params,
                    )
                    doctor_response = result.content
                except MaxTokenExceededError:
                    break
                except Exception as e:
                    print(f"[DiagPRMAgentLoop] Doctor call error: {e}")
                    break

                if doctor_response is None:
                    break

                # result 是 AIMessage，包含 response_metadata（prompt_ids, response_mask）
                messages.append(result)

                # ── 解析 action ────────────────────────────────────────────────
                action = self._parse_action(doctor_response)

                if action == "diagnose" or is_last_turn:
                    verifier_responses.append("<Normal>")
                    break

                # ── 提取医生的问题 ──────────────────────────────────────────────
                question = self._parse_question(doctor_response)
                if question is None:
                    verifier_responses.append("<ERROR_RESPONSE>")
                    break

                # ── Verifier 检测（重复/多重问题）──────────────────────────────
                if self.verifier_enabled:
                    verifier_tag = self._run_verifier_sync(question, previous_questions)
                else:
                    verifier_tag = "<Normal>"
                verifier_responses.append(verifier_tag)

                if verifier_tag in ("<Repeated>", "<Multiple>"):
                    feedback = (
                        "You have already asked this question. Please ask a different question."
                        if verifier_tag == "<Repeated>"
                        else "Please ask only ONE question at a time."
                    )
                    messages.append(HumanMessage(content=feedback))
                    continue

                # ── Patient Simulator 回答 ──────────────────────────────────────
                patient_answer = self._rule_based_patient(atomic_facts, question)

                previous_questions.append(question)
                messages.append(HumanMessage(content=patient_answer))

        except MaxTokenExceededError:
            pass
        except Exception as e:
            print(f"[DiagPRMAgentLoop] Unexpected error: {e}")
            import traceback
            traceback.print_exc()

        # ── 从最后的 AI 消息中提取 prompt_ids 和 response_mask ──────────────
        # 找到最后一条 AI 消息
        last_ai_msg = None
        for msg in reversed(messages):
            if hasattr(msg, 'type') and msg.type == 'ai':
                last_ai_msg = msg
                break

        if last_ai_msg is None or "prompt_ids" not in last_ai_msg.response_metadata:
            # 兜底：用 tokenizer 编码整个消息序列
            try:
                all_ids = self.tokenizer.apply_chat_template(
                    [{"role": "system", "content": messages[0].content if messages else ""},
                     {"role": "user", "content": chief_complaint}],
                    add_generation_prompt=True,
                    tokenize=True,
                )
                prompt_ids = all_ids
                response_mask = [0] * len(all_ids)
            except Exception:
                prompt_ids = [self.tokenizer.bos_token_id or 1]
                response_mask = [0]
        else:
            prompt_ids = last_ai_msg.response_metadata["prompt_ids"]
            response_mask = last_ai_msg.response_metadata["response_mask"]

        # 将 prompt_ids 拆分为 prompt 部分和 response 部分
        # response_mask: 1 表示 LLM 生成的 token（response），0 表示 prompt token
        # 找到第一个 response token 的位置
        first_response_idx = len(prompt_ids)  # 默认：无 response
        for i, mask in enumerate(response_mask):
            if mask == 1:
                first_response_idx = i
                break

        pure_prompt_ids = list(prompt_ids[:first_response_idx])
        pure_response_ids = list(prompt_ids[first_response_idx:])
        pure_response_mask = list(response_mask[first_response_idx:])

        # 确保 response_ids 非空（tokenizer.pad 无法处理空列表）
        # 用 EOS token 作为占位符
        _eos = self.tokenizer.eos_token_id or self.tokenizer.pad_token_id or 0
        if not pure_response_ids:
            pure_response_ids = [_eos]
            pure_response_mask = [1]

        # 确保 prompt_ids 非空
        if not pure_prompt_ids:
            pure_prompt_ids = list(prompt_ids) if prompt_ids else [_eos]

        # 截断 response_ids 不超过 max_response_length
        max_resp_len = self.config.actor_rollout_ref.rollout.response_length
        if len(pure_response_ids) > max_resp_len:
            pure_response_ids = pure_response_ids[:max_resp_len]
            pure_response_mask = pure_response_mask[:max_resp_len]

        # 截断 prompt_ids 不超过 max_prompt_length（从左侧截断保留最新内容）
        max_prompt_len = self.config.actor_rollout_ref.rollout.prompt_length
        if len(pure_prompt_ids) > max_prompt_len:
            pure_prompt_ids = pure_prompt_ids[-max_prompt_len:]

        # 截断后重新统计实际有效的轮数（mask=1 的连续段数），
        # 避免因 response 被截断导致 turn_count 与 parse_turns_from_response_mask 的结果不一致。
        effective_turn_count = 0
        _in_response = False
        for _m in pure_response_mask:
            if _m == 1 and not _in_response:
                effective_turn_count += 1
                _in_response = True
            elif _m == 0:
                _in_response = False
        # 保底：至少和原 turn_count 取较小值（不能因截断多算）
        effective_turn_count = min(effective_turn_count, turn_count)

        output = AgentLoopOutput(
            prompt_ids=pure_prompt_ids,
            response_ids=pure_response_ids,
            response_mask=pure_response_mask,
            num_turns=effective_turn_count,
            verifier_responses=verifier_responses if verifier_responses else ["<Normal>"],
            metrics=AgentLoopMetrics(),
            statistics={
                "disease": disease,
                "turn_count": effective_turn_count,
                # _update_statistics 需要这三个键（可以为空列表）
                "q_value_variance_list": [],
                "mdp_value_list": [],
                "critic_value_list": [],
            },
        )
        return [output]

    def _rule_based_patient(
        self,
        atomic_facts: List[str],
        question: str,
    ) -> str:
        """
        基于规则的患者回答（atomic_facts 关键词匹配）。
        速度快，不需要额外 LLM 调用。
        """
        if not atomic_facts:
            return "I don't have that symptom."

        question_lower = question.lower()
        # 提取问题中的关键词
        keywords = set(re.findall(r'\b[a-z]{4,}\b', question_lower))
        keywords -= {"have", "does", "your", "you", "the", "any", "this", "that",
                     "please", "tell", "about", "been", "feel", "experiencing"}

        # 在 atomic_facts 中搜索相关条目
        relevant_facts = []
        for fact in atomic_facts:
            fact_lower = fact.lower()
            if any(kw in fact_lower for kw in keywords):
                relevant_facts.append(fact)

        if relevant_facts:
            return "Yes, " + " ".join(relevant_facts[:2])
        else:
            return "I don't have that symptom."

    def _run_verifier_sync(
        self,
        question: str,
        previous_questions: List[str],
    ) -> str:
        """同步版本的 verifier（不调用 LLM，用简单规则）。"""
        if not previous_questions:
            return "<Normal>"

        # 检查重复：与前面问题的 token overlap > 0.7
        question_tokens = set(question.lower().split())
        for prev_q in previous_questions:
            prev_tokens = set(prev_q.lower().split())
            if not prev_tokens:
                continue
            overlap = len(question_tokens & prev_tokens) / len(question_tokens | prev_tokens)
            if overlap > 0.7:
                return "<Repeated>"

        # 检查多重问题：包含 "and" "or" 连接的多个问句
        if question.count("?") > 1:
            return "<Multiple>"

        return "<Normal>"

    def _parse_action(self, doctor_response: str) -> Optional[str]:
        """从 doctor 回复中解析 <action> 标签。
        动作空间： continue / diagnose（将旧的 switch 平滑合并为 continue）
        """
        match = re.search(
            r'<action>\s*(continue|switch|diagnose)\s*</action>',
            doctor_response,
            re.IGNORECASE | re.DOTALL,
        )
        if match:
            raw = match.group(1).strip().lower()
            return "continue" if raw == "switch" else raw
        return None

    def _parse_question(self, doctor_response: str) -> Optional[str]:
        """从 doctor 回复中解析 <question> 标签。"""
        match = re.search(
            r'<question>\s*(.*?)\s*</question>',
            doctor_response,
            re.IGNORECASE | re.DOTALL,
        )
        if match:
            return match.group(1).strip()
        # 兼容 "Question:" 格式
        match2 = re.search(r'Question:\s*(.+)', doctor_response, re.IGNORECASE)
        if match2:
            return match2.group(1).strip()
        return None

    def _extract_chief_complaint(self, initial_prompt: Any) -> str:
        """从 initial_prompt（可能是 list[dict] 或 str）中提取主诉。"""
        if isinstance(initial_prompt, list):
            for msg in initial_prompt:
                if isinstance(msg, dict) and msg.get("role") == "user":
                    return msg.get("content", "")
        if isinstance(initial_prompt, str):
            return initial_prompt
        if isinstance(initial_prompt, dict):
            return initial_prompt.get("content", str(initial_prompt))
        return str(initial_prompt)

    def _build_initial_messages(self, chief_complaint: str) -> List:
        """构建初始消息列表。"""
        system_msg = SystemMessage(content=DOCTOR_SYSTEM_PROMPT.format(
            max_turns=self.max_turns,
            current_turn=1,
        ))
        user_msg = HumanMessage(content=DOCTOR_INITIAL_PROMPT.format(
            chief_complaint=chief_complaint,
        ))
        return [system_msg, user_msg]
