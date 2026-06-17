"""
DiagPRM - Multi-turn Diagnostic Dialogue Agent Loop

Implements the async multi-turn diagnostic dialogue between the Doctor Agent and Patient Simulator.

Dialogue protocol (each turn):
  Doctor  : <think>...</think><hypothesis name="..."><confirmed>...</confirmed></hypothesis>
             <action>continue|diagnose</action>
             <question>...</question>  or  <diagnosis>...</diagnosis>
  Patient : Answers the doctor's question based on atomic_facts via LLM API call

Doctor Agent prompt format (state-summary mode):
  System prompt : role description + output format (DOCTOR_SYSTEM_PROMPT, updated each turn with current_turn)
  User message  : rebuilt each turn, no full history retained
    - Turn 1  : DOCTOR_INITIAL_PROMPT (chief complaint, no confirmed symptoms)
    - Turn N  : DOCTOR_TURN_PROMPT (chief complaint + current hypothesis + accumulated symptoms + last new finding)
  No AIMessage history appended: each turn input is exactly [System, User], fixed context length

State tracking (Agent Loop side):
  current_hypothesis   : parsed from <hypothesis> in previous Doctor output
  collected_symptoms   : accumulated confirmed symptoms from Patient answers
  new_finding          : new symptoms added this turn (or "No new findings" for invalid questions)

Patient Simulator:
  - Uses atomic_facts from the dataset
  - Calls an external OpenAI-compatible API (patient_api_base / patient_model config)
  - Falls back to rule-based matching if the API call fails
"""

import re
import asyncio
import json
import os
from typing import Any, Dict, List, Optional, Tuple
import aiohttp
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, convert_to_messages

from recipe.atpo.agent_loop import AgentLoopBase, AgentLoopOutput, AgentLoopMetrics
from recipe.atpo.chat_model import ChatModel, MaxTokenExceededError
from recipe.diagprm.prompts import (
    DOCTOR_SYSTEM_PROMPT,
    DOCTOR_INITIAL_PROMPT,
    DOCTOR_TURN_PROMPT,
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
      - Doctor Agent 的结构化 CoT 格式（<hypothesis name="...">, <action>）
      - Patient Simulator（基于 atomic_facts 的规则 + LLM 混合问答）
      - Verifier（检测重复/多重问题）
      - 最大轮次控制

    Config (from agent.yaml):
      max_turns           : max dialogue turns (default: 10)
      verifier_enabled    : enable repeated-question detection (default: True)
      patient_api_base    : base URL of the OpenAI-compatible API for the patient simulator
                            (default: PATIENT_API_BASE env var, or "http://localhost:8001/v1")
      patient_model       : model name for the patient simulator
                            (default: PATIENT_MODEL env var, or "gpt-4o-mini")
      patient_max_tokens  : max tokens for patient responses (default: 256)
    """

    def __init__(self, trainer_config, server_manager, tokenizer, processor, **kwargs):
        super().__init__(trainer_config, server_manager, tokenizer, processor, **kwargs)
        self.max_turns = kwargs.get("max_turns", 10)
        self.verifier_enabled = kwargs.get("verifier_enabled", True)
        # Patient LLM API config
        self.patient_api_base = kwargs.get(
            "patient_api_base",
            os.environ.get("PATIENT_API_BASE", "http://localhost:8001/v1"),
        )
        self.patient_model = kwargs.get(
            "patient_model",
            os.environ.get("PATIENT_MODEL", "gpt-4o-mini"),
        )
        self.patient_max_tokens = kwargs.get("patient_max_tokens", 256)

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
        # 当 data.return_raw_chat=True 时，dataset 把原始消息存在 "raw_prompt" 字段；
        # 否则存在 "prompt" 字段（JSON 字符串形式）。两个都尝试。
        raw_prompt = kwargs.get("raw_prompt") or kwargs.get("prompt", [])
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

        # New schema: disease / chief_complaint / symptoms_pool / initial_symptoms
        # are top-level fields in the dataset row (not nested under reward_model).
        # Fall back to the old reward_model.ground_truth layout for compatibility.
        disease = (
            kwargs.get("disease")
            or ground_truth.get("disease", "unknown")
        )
        # initial_symptoms: KG verbatim strings revealed in the chief complaint (turn-0).
        # Used only to build chief_complaint; NOT the Patient's full fact sheet.
        initial_symptoms = (
            kwargs.get("initial_symptoms")
            or ground_truth.get("initial_symptoms", [])
        )

        # Patient's full fact sheet = complete symptoms_pool (all KG verbatim keys).
        # This allows the Patient to answer ANY symptom the Doctor asks about,
        # not just the 1-3 revealed in the chief complaint.
        # <fact> will be a verbatim copy of a symptoms_pool key -> direct KG match.
        symptoms_pool = (
            kwargs.get("symptoms_pool")
            or ground_truth.get("symptoms_pool", {})
        )
        if symptoms_pool:
            # Use all symptom keys (KG verbatim strings) as the Patient's fact sheet
            atomic_facts = list(symptoms_pool.keys())
        else:
            # Back-compat: old format stored "atomic_facts" as sentences
            atomic_facts_legacy = ground_truth.get("atomic_facts", [])
            atomic_facts = initial_symptoms if initial_symptoms else atomic_facts_legacy

        # chief_complaint: prefer top-level field (new schema), else parse from prompt
        chief_complaint = (
            kwargs.get("chief_complaint")
            or self._extract_chief_complaint(raw_prompt)
        )

        # ── 状态摘要模式：系统侧维护对话状态，每轮重建 [System, User] ──────────
        # 不保留完整对话历史，context 长度固定为 2 条消息
        verifier_responses = []
        previous_questions = []
        turn_count = 0

        # 系统侧维护的对话状态
        collected_symptoms: List[str] = []   # 累积确认的症状（原子事实，KG 匹配用）
        current_hypothesis: str = ""         # 上一轮 Doctor 输出的假设名
        last_ai_msg = None                   # 保留最后一条 AIMessage（用于提取 token ids）
        dialogue_history: List[dict] = []    # 每轮 {turn_id, doctor_response, patient_answer, fact}
        new_finding: str = ""               # Bug3 fix: 初始化防止第 2 轮 NameError

        try:
            for turn_idx in range(self.max_turns):
                turn_count = turn_idx + 1
                is_last_turn = (turn_idx == self.max_turns - 1)

                # ── 构建本轮 [System, User] 消息 ───────────────────────────────
                system_msg = SystemMessage(content=DOCTOR_SYSTEM_PROMPT.format(
                    max_turns=self.max_turns,
                    current_turn=turn_count,
                ))
                if turn_idx == 0:
                    # 第 1 轮：只有主诉，无历史状态
                    user_msg = HumanMessage(content=DOCTOR_INITIAL_PROMPT.format(
                        chief_complaint=chief_complaint,
                    ))
                else:
                    # 第 N 轮：注入累积状态摘要
                    confirmed_str = ", ".join(collected_symptoms) if collected_symptoms else "None"
                    user_msg = HumanMessage(content=DOCTOR_TURN_PROMPT.format(
                        chief_complaint=chief_complaint,
                        hypothesis=current_hypothesis or "Undetermined",
                        confirmed_symptoms=confirmed_str,
                        new_finding=new_finding,
                    ))
                messages = [system_msg, user_msg]

                # ── Doctor 生成回复 ────────────────────────────────────────────
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

                last_ai_msg = result  # 保留用于 token 提取

                # ── 解析 action ────────────────────────────────────────────────
                action = self._parse_action(doctor_response)

                # ── 更新系统侧的当前假设（从本轮输出解析）────────────────────
                parsed_hyp, _ = self._parse_hypothesis_name(doctor_response)
                if parsed_hyp:
                    current_hypothesis = parsed_hyp

                if action == "diagnose" or is_last_turn:
                    verifier_responses.append("<Normal>")
                    # 记录最终轮（诊断轮无 patient 回答）
                    dialogue_history.append({
                        "turn_id": turn_idx,
                        "doctor_response": doctor_response,
                        "patient_answer": "",
                        "patient_fact": "",
                        "is_final": True,
                    })
                    break

                # ── 提取医生的问题 ─────────────────────────────────────────────
                question = self._parse_question(doctor_response)
                if question is None:
                    verifier_responses.append("<ERROR_RESPONSE>")
                    new_finding = "Last question was malformed (no <question> tag found)."
                    break

                # ── Verifier 检测（重复/多重问题）─────────────────────────────
                if self.verifier_enabled:
                    verifier_tag = self._run_verifier_sync(question, previous_questions)
                else:
                    verifier_tag = "<Normal>"
                verifier_responses.append(verifier_tag)

                if verifier_tag == "<Repeated>":
                    new_finding = "Last question was invalid: you already asked this question. No new information was obtained."
                    # Bug2 fix: verifier 拦截时也需要追加问题，否则下轮仍能通过
                    previous_questions.append(question)
                    # 记录本轮（patient_fact=unknown 表示无效提问）
                    dialogue_history.append({
                        "turn_id": turn_idx,
                        "doctor_response": doctor_response,
                        "patient_answer": "",
                        "patient_fact": "unknown",
                    })
                    continue
                elif verifier_tag == "<Multiple>":
                    new_finding = "Last question was invalid: you asked multiple questions at once. No new information was obtained."
                    previous_questions.append(question)
                    dialogue_history.append({
                        "turn_id": turn_idx,
                        "doctor_response": doctor_response,
                        "patient_answer": "",
                        "patient_fact": "unknown",
                    })
                    continue

                # ── Patient Simulator 回答（LLM API）──────────────────────────
                patient_raw = await self._llm_patient(atomic_facts, question)

                # 解析 patient 回复中的 <answer> 和 <fact> 字段
                patient_answer = self._parse_patient_answer(patient_raw)
                patient_fact   = self._parse_patient_fact(patient_raw)

                # ── Update accumulated symptoms, generate new_finding ──────────
                # new_finding 传给 Doctor 的是自然语言 <answer>（真实性）
                # patient_fact 传给 Reward Manager 用于 KG 匹配
                if patient_fact and patient_fact.lower() != "unknown":
                    if patient_fact not in collected_symptoms:
                        collected_symptoms.append(patient_fact)
                    new_finding = f"Patient answer: \"{patient_answer}\""
                else:
                    new_finding = f"Patient answer: \"{patient_answer}\" — no relevant symptom found."

                # Bug3 fix: previous_questions 在 verifier 拦截时也要追加，避免死循环
                previous_questions.append(question)

                # 记录本轮对话（供 Reward Manager 使用）
                dialogue_history.append({
                    "turn_id": turn_idx,
                    "doctor_response": doctor_response,
                    "patient_answer": patient_answer,
                    "patient_fact": patient_fact,  # 原子事实或 "unknown"
                })

        except MaxTokenExceededError:
            pass
        except Exception as e:
            print(f"[DiagPRMAgentLoop] Unexpected error: {e}")
            import traceback
            traceback.print_exc()

        # ── 从最后的 AI 消息中提取 prompt_ids 和 response_mask ──────────────
        # 状态摘要模式下 last_ai_msg 在循环中已维护，无需再遍历 messages
        if last_ai_msg is None or "prompt_ids" not in last_ai_msg.response_metadata:
            # 兜底：用 tokenizer 编码最后一轮的 [system, user] 消息
            try:
                all_ids = self.tokenizer.apply_chat_template(
                    [{"role": "system", "content": DOCTOR_SYSTEM_PROMPT.format(
                        max_turns=self.max_turns, current_turn=turn_count)},
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
                # 完整对话轨迹：供 Reward Manager 直接计算 turn-level reward
                # 每条记录: {turn_id, doctor_response, patient_answer, patient_fact, [is_final]}
                "dialogue_history": dialogue_history,
                # _update_statistics 需要这三个键（可以为空列表）
                "q_value_variance_list": [],
                "mdp_value_list": [],
                "critic_value_list": [],
            },
        )
        return [output]

    async def _llm_patient(
        self,
        atomic_facts: List[str],
        question: str,
    ) -> str:
        """
        LLM-based patient simulator.

        Calls an external OpenAI-compatible chat API with PATIENT_SYSTEM_PROMPT
        (containing the atomic facts) and the doctor's question as the user message.
        Falls back to a simple rule-based answer if the API call fails.
        """
        # atomic_facts is now a list of KG verbatim symptom strings.
        # Format them as bullet points for the Patient Simulator prompt.
        facts_text = "\n".join(f"- {f}" for f in atomic_facts) if atomic_facts else "(no known facts)"
        system_content = PATIENT_SYSTEM_PROMPT.format(atomic_facts=facts_text)

        payload = {
            "model": self.patient_model,
            "messages": [
                {"role": "system", "content": system_content},
                {"role": "user", "content": question},
            ],
            "max_tokens": self.patient_max_tokens,
            "temperature": 0.0,
        }
        headers = {"Content-Type": "application/json"}
        api_key = os.environ.get("PATIENT_API_KEY", "")
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        url = self.patient_api_base.rstrip("/") + "/chat/completions"
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url,
                    json=payload,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=30),
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        return data["choices"][0]["message"]["content"].strip()
                    else:
                        text = await resp.text()
                        print(f"[Patient API] HTTP {resp.status}: {text[:200]}")
        except Exception as e:
            print(f"[Patient API] call failed: {e}")

        # Fallback: simple keyword match against atomic_facts
        return self._rule_based_patient_fallback(atomic_facts, question)

    def _rule_based_patient_fallback(
        self,
        atomic_facts: List[str],
        question: str,
    ) -> str:
        """Fallback rule-based patient answer used when the LLM API is unavailable."""
        question_lower = question.lower()
        keywords = set(re.findall(r'\b[a-z]{4,}\b', question_lower))
        keywords -= {"have", "does", "your", "you", "the", "any", "this", "that",
                     "please", "tell", "about", "been", "feel", "experiencing",
                     "patient", "currently", "present", "symptom", "condition"}
        if atomic_facts and keywords:
            for fact in atomic_facts:
                if any(kw in fact.lower() for kw in keywords):
                    return "Yes, " + fact
        return "I'm not sure about that."

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

    def _parse_hypothesis_name(self, doctor_response: str) -> Tuple[Optional[str], Optional[str]]:
        """从 doctor 回复中解析 <hypothesis> 的假设名称和 action。

        Returns:
            (hypothesis_name | None, action | None)
        """
        hypothesis = None
        action = None

        hyp_match = re.search(
            r'<hypothesis[^>]+name\s*=\s*["\']([^"\']+)["\']',
            doctor_response,
            re.IGNORECASE,
        )
        if hyp_match:
            hypothesis = hyp_match.group(1).strip()

        action_match = re.search(
            r'<action>\s*(continue|switch|diagnose)\s*</action>',
            doctor_response,
            re.IGNORECASE | re.DOTALL,
        )
        if action_match:
            raw = action_match.group(1).strip().lower()
            action = "continue" if raw == "switch" else raw

        return hypothesis, action

    def _parse_patient_answer(self, raw: str) -> str:
        """Extract the <answer>...</answer> field from patient LLM output."""
        match = re.search(r'<answer>\s*(.*?)\s*</answer>', raw, re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1).strip()
        # Fallback: return the whole response if no tag found
        return raw.strip()

    def _parse_patient_fact(self, raw: str) -> str:
        """Extract the <fact>...</fact> field from patient LLM output.

        Returns the atomic fact string, or 'unknown' if not present / explicitly unknown.
        """
        match = re.search(r'<fact>\s*(.*?)\s*</fact>', raw, re.IGNORECASE | re.DOTALL)
        if match:
            fact = match.group(1).strip()
            return fact if fact else "unknown"
        return "unknown"

    def _is_negative_answer(self, answer: str) -> bool:
        """Return True if the patient's answer is a denial (no new symptom confirmed)."""
        deny_pattern = re.compile(
            r'^\s*(no|not|i don\'t|i\'m not|i do not|i have not|i\'m not sure|i don\'t have|i do not have)',
            re.IGNORECASE,
        )
        return bool(deny_pattern.match(answer.strip()))

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
