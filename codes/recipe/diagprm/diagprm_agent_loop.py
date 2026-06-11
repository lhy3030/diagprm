"""
DiagPRM - Multi-turn Diagnostic Dialogue Agent Loop

实现 Doctor Agent 与 Patient Simulator 的异步多轮对话。

对话协议（每轮）：
  Doctor  : <think>...</think><hypothesis_state>...</hypothesis_state>
             <action>continue|switch|diagnose</action>
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
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from recipe.atpo.agent_loop import AgentLoopBase, AgentLoopOutput
from recipe.atpo.api_request_async import request_vllm_async
from recipe.atpo.chat_model import ChatModel, MaxTokenExceededError, convert_to_agent_output
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

    def __init__(self, config: Dict, **kwargs):
        super().__init__(config, **kwargs)
        self.max_turns = config.get("max_turns", 10)
        self.use_llm_patient = config.get("use_llm_patient", True)
        self.verifier_enabled = config.get("verifier_enabled", True)
        self.doctor_model = None  # 由 AgentLoopBase 的 set_model 设置

    async def run(
        self,
        chat_model: ChatModel,
        initial_prompt: Dict,
        ground_truth: Any = None,
        extra_info: Dict = None,
        **kwargs,
    ) -> AgentLoopOutput:
        """
        执行一轮完整的多轮诊断对话。

        Args:
            chat_model    : verl 的 ChatModel 接口（调用 vLLM）
            initial_prompt: 数据集中的 prompt（通常是主诉）
            ground_truth  : {disease: str, atomic_facts: List[str]}
            extra_info    : 数据集的 extra_info 字段

        Returns:
            AgentLoopOutput：包含完整对话序列和 metadata
        """
        # 获取患者信息
        if ground_truth is None:
            ground_truth = {}
        disease = ground_truth.get("disease", "unknown")
        atomic_facts = ground_truth.get("atomic_facts", [])
        chief_complaint = self._extract_chief_complaint(initial_prompt)

        # 初始化对话历史
        messages = self._build_initial_messages(chief_complaint)
        verifier_responses = []
        human_responses_log = []
        previous_questions = []
        turn_count = 0

        try:
            for turn_idx in range(self.max_turns):
                turn_count = turn_idx + 1
                is_last_turn = (turn_idx == self.max_turns - 1)

                # ── Doctor 生成回复 ────────────────────────────────────────────
                # 更新系统提示中的 current_turn
                messages[0] = SystemMessage(content=DOCTOR_SYSTEM_PROMPT.format(
                    max_turns=self.max_turns,
                    current_turn=turn_count,
                ))

                doctor_response = await self._call_doctor(chat_model, messages)
                if doctor_response is None:
                    # 超长或错误，强制结束
                    break

                messages.append(AIMessage(content=doctor_response))

                # ── 解析 action ────────────────────────────────────────────────
                action = self._parse_action(doctor_response)

                if action == "diagnose" or is_last_turn:
                    # Episode 结束
                    verifier_responses.append("<Normal>")  # 诊断轮无 verifier
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
                    # 不调用患者，给出格式提示并继续
                    feedback = (
                        "You have already asked this question. Please ask a different question."
                        if verifier_tag == "<Repeated>"
                        else "Please ask only ONE question at a time."
                    )
                    messages.append(HumanMessage(content=feedback))
                    human_responses_log.append(feedback)
                    continue

                # ── Patient Simulator 回答 ──────────────────────────────────────
                if self.use_llm_patient:
                    patient_answer = await self._call_patient(
                        chat_model, atomic_facts, question
                    )
                else:
                    patient_answer = self._rule_based_patient(atomic_facts, question)

                if patient_answer is None:
                    patient_answer = "I'm not sure about that."

                previous_questions.append(question)
                human_responses_log.append(patient_answer)
                messages.append(HumanMessage(content=patient_answer))

        except MaxTokenExceededError:
            pass
        except Exception as e:
            print(f"[DiagPRMAgentLoop] Error: {e}")

        # 转换为 verl AgentLoopOutput 格式
        return self._build_output(
            messages=messages,
            verifier_responses=verifier_responses,
            turn_count=turn_count,
            ground_truth=ground_truth,
        )

    async def _call_doctor(
        self,
        chat_model: ChatModel,
        messages: List,
    ) -> Optional[str]:
        """调用 Doctor Agent 模型生成回复。"""
        try:
            result = await chat_model.ainvoke(messages)
            return convert_to_agent_output(result)
        except MaxTokenExceededError:
            raise
        except Exception as e:
            print(f"[Doctor call error] {e}")
            return None

    async def _call_patient(
        self,
        chat_model: ChatModel,
        atomic_facts: List[str],
        question: str,
    ) -> Optional[str]:
        """调用 Patient Simulator LLM 生成患者回答。"""
        facts_text = "\n".join(f"- {f}" for f in atomic_facts) if atomic_facts else "No additional information."
        patient_messages = [
            SystemMessage(content=PATIENT_SYSTEM_PROMPT.format(atomic_facts=facts_text)),
            HumanMessage(content=f"Doctor's question: {question}"),
        ]
        try:
            # 患者模拟器用较低温度（更确定性）
            result = await chat_model.ainvoke(
                patient_messages,
                temperature=0.3,
                max_tokens=200,
            )
            return convert_to_agent_output(result)
        except Exception as e:
            print(f"[Patient call error] {e}")
            return None

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
        """从 doctor 回复中解析 <action> 标签。"""
        match = re.search(
            r'<action>\s*(continue|switch|diagnose)\s*</action>',
            doctor_response,
            re.IGNORECASE | re.DOTALL,
        )
        if match:
            return match.group(1).strip().lower()
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

    def _build_output(
        self,
        messages: List,
        verifier_responses: List[str],
        turn_count: int,
        ground_truth: Dict,
    ) -> AgentLoopOutput:
        """将对话历史转换为 verl 格式的 AgentLoopOutput。"""
        return AgentLoopOutput(
            messages=messages,
            metadata={
                "verifier_responses": verifier_responses,
                "__num_turns__": turn_count,
                "ground_truth": ground_truth,
            },
        )
