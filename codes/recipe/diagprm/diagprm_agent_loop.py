"""
DiagPRM - Multi-turn Diagnostic Dialogue Agent Loop

Implements the async multi-turn diagnostic dialogue between the Doctor Agent and Patient Simulator.

Dialogue protocol (each turn):
  Doctor  : JSON block with keys: action, question/diagnosis
  Patient : JSON block with keys: answer, fact_id

Doctor Agent prompt format (full-history multi-turn mode):
  System prompt : role description + output format (DOCTOR_SYSTEM_PROMPT, updated each turn with current_turn)
  History       : previous patient messages, Doctor JSON responses, KG feedback, and patient answers
  Next turn     : current prompt is the previous prompt + previous Doctor response + previous patient answer

State tracking (Agent Loop side):
  collected_symptoms   : accumulated confirmed symptoms from Patient answers
  new_finding          : new symptoms added this turn (or "No new findings" for invalid questions)

Patient Simulator:
  - Uses hidden symptom facts from the dataset
  - Calls an external OpenAI-compatible API (patient_api_base / patient_model config)
  - Falls back to rule-based matching if the API call fails
"""

import re
import asyncio
import json
import os
from contextlib import nullcontext
from typing import Any, Dict, List, Optional, Tuple
import aiohttp
from langchain_core.messages import HumanMessage, SystemMessage

# 全局信号量：限制同时发往外部 Patient API 的并发请求数，避免 429 rate limit
# 通过环境变量 PATIENT_API_CONCURRENCY 控制，默认 3
_PATIENT_API_SEM: Optional[asyncio.Semaphore] = None

def _get_patient_sem() -> asyncio.Semaphore:
    global _PATIENT_API_SEM
    if _PATIENT_API_SEM is None:
        limit = int(os.environ.get("PATIENT_API_CONCURRENCY", "3"))
        _PATIENT_API_SEM = asyncio.Semaphore(limit)
    return _PATIENT_API_SEM


def _strip_thinking_text(text: str) -> str:
    """Remove Qwen-style thinking traces from patient simulator outputs."""
    if not text:
        return ""
    cleaned = re.sub(r'<think>[\s\S]*?</think>', '', text, flags=re.IGNORECASE)
    lower = cleaned.lower()
    if "</think>" in lower:
        cleaned = cleaned[lower.rfind("</think>") + len("</think>"):]
    elif "<think>" in lower:
        cleaned = cleaned[:lower.find("<think>")]
    return cleaned.strip()

from recipe.atpo.agent_loop import AgentLoopBase, AgentLoopOutput, AgentLoopMetrics
from recipe.atpo.chat_model import ChatModel, MaxTokenExceededError
from recipe.diagprm.kg_utils import (
    load_kg,
    lookup_disease_name,
    query_diseases_by_symptom,
    query_symptoms_by_disease,
)
from recipe.diagprm.prompts import (
    DOCTOR_SYSTEM_PROMPT,
    DOCTOR_SYSTEM_PROMPT_NO_KG,
    DOCTOR_FINAL_DIAGNOSIS_PROMPT,
    PATIENT_SYSTEM_PROMPT,
    PATIENT_OPENING_PROMPT,
)


# ──────────────────────────────────────────────────────────────────────────────
# DiagPRM Agent Loop
# ──────────────────────────────────────────────────────────────────────────────

class DiagPRMAgentLoop(AgentLoopBase):
    """
    DiagPRM 多轮诊断对话 Agent Loop。
    
    继承自 ATPO 的 AgentLoopBase，重写对话逻辑以支持：
      - Doctor Agent 的干净 JSON 动作格式（ask / diagnose）
      - Patient Simulator（基于 hidden symptom facts 的规则 + LLM 混合问答）
      - Verifier（检测重复/多重问题）
      - 最大轮次控制

    Config (from agent.yaml):
      max_turns           : max dialogue turns (default: 10)
      verifier_enabled    : enable repeated-question detection (default: True)
      patient_api_base    : base URL of the OpenAI-compatible API for the patient simulator
                            (default: PATIENT_API_BASE env var, or "http://localhost:8001/v1")
      patient_model       : model name for the patient simulator
                            (default: PATIENT_MODEL env var, or "gpt-4o-mini")
      patient_max_tokens  : max tokens for patient responses (default: 512)
      kg_tool_enabled     : whether Doctor can observe KG query feedback (default: False)
      kg_path             : path to master_kg.json (for KG query tool; falls back to
                            reward_model.kg_path in trainer config if not provided)
      force_diagnose_on_last_turn : force final turn to diagnose and retry once
                                    if the model still asks (default: True)
    """

    def __init__(self, trainer_config, server_manager, tokenizer, processor, **kwargs):
        super().__init__(trainer_config, server_manager, tokenizer, processor, **kwargs)
        self.max_turns = int(
            os.environ.get("MAX_TURNS_OVERRIDE")
            or kwargs.get("max_turns", 10)
        )
        self.verifier_enabled = kwargs.get("verifier_enabled", True)
        # Patient LLM API config
        # 优先级：环境变量 > yaml/kwargs > 默认值
        # 环境变量允许在运行时动态切换（如本地 vLLM 替换外网 API），无需修改 yaml
        self.patient_api_base = (
            os.environ.get("PATIENT_API_BASE")
            or kwargs.get("patient_api_base", "http://localhost:8001/v1")
        )
        self.patient_model = (
            os.environ.get("PATIENT_MODEL")
            or kwargs.get("patient_model", "gpt-4o-mini")
        )
        self.patient_max_tokens = int(
            os.environ.get("PATIENT_MAX_TOKENS")
            or kwargs.get("patient_max_tokens", 512)
        )
        # 是否开启思考模式（内部模型如 qwen3.5-plus 支持 enable_thinking 关闭，节省 token）
        self.patient_enable_thinking = bool(kwargs.get("patient_enable_thinking", False))
        self.kg_tool_enabled = bool(kwargs.get("kg_tool_enabled", False))
        self.force_diagnose_on_last_turn = (
            os.environ.get("FORCE_DIAGNOSE_ON_LAST_TURN", "1").strip().lower()
            not in {"0", "false", "no", "off"}
        )
        self.doctor_system_prompt = (
            DOCTOR_SYSTEM_PROMPT if self.kg_tool_enabled else DOCTOR_SYSTEM_PROMPT_NO_KG
        )

        # KG tool: 加载知识图谱供医生诊断时查询 verbatim 疾病名
        kg_path = (
            kwargs.get("kg_path")
            or os.environ.get("KG_PATH", "")
            or getattr(getattr(self.config, "reward_model", None), "kg_path", "")
        )
        if self.kg_tool_enabled and kg_path:
            try:
                self._kg = load_kg(kg_path)
                print(f"[DiagPRMAgentLoop] KG loaded: {len(self._kg):,} diseases")
            except Exception as e:
                print(f"[DiagPRMAgentLoop] WARNING: failed to load KG from {kg_path}: {e}")
                self._kg = None
        elif self.kg_tool_enabled:
            print("[DiagPRMAgentLoop] WARNING: kg_path not set; KG query tool disabled.")
            self._kg = None
        else:
            self._kg = None

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

        # reward_model 包含 ground_truth（disease, symptoms_pool / symptom_facts）
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

        # Patient's full private fact sheet = complete symptoms_pool (all KG verbatim
        # keys). We expose ids to the Patient simulator and keep the id->text map
        # hidden from the Doctor. The Reward Manager resolves patient_fact_id using
        # this map, so Doctor only sees natural-language answers.
        symptoms_pool = (
            kwargs.get("symptoms_pool")
            or ground_truth.get("symptoms_pool", {})
        )
        symptom_facts = self._build_symptom_facts(symptoms_pool, initial_symptoms, ground_truth)
        fact_id_to_text = {f["fact_id"]: f["text"] for f in symptom_facts}
        fact_text_to_id = {f["text"]: f["fact_id"] for f in symptom_facts}
        # chief_complaint: prefer top-level field (new schema), else parse from prompt
        chief_complaint = (
            kwargs.get("chief_complaint")
            or self._extract_chief_complaint(raw_prompt)
        )

        # ── 标准 multi-turn 模式：维护完整对话历史列表 ──────────────────────
        verifier_responses = []
        previous_questions = []
        turn_count = 0

        # 系统侧维护（仅用于 reward 计算，不影响 doctor prompt）
        collected_symptoms: List[str] = []   # 累积确认的症状（KG 匹配用）
        last_ai_msg = None                   # 保留最后一条 AIMessage（用于提取 token ids）
        dialogue_history: List[dict] = []    # 每轮 {turn_id, doctor_response, patient_answer, patient_fact_id}

        # ── 第 0 步：Patient Agent 生成自然语言主诉开场白 ────────────────────
        patient_opening = await self._llm_patient_opening(initial_symptoms)

        # ── 初始化完整对话历史 ───────────────────────────────────────────────
        # messages 列表结构：
        #   [0] SystemMessage (system prompt，每轮更新 current_turn)
        #   [1] HumanMessage  (patient 主诉)
        #   [2] AIMessage     (doctor 第1轮回复)
        #   [3] HumanMessage  (patient 第1轮回答)
        #   [4] AIMessage     (doctor 第2轮回复)
        #   ...
        messages: List = [
            SystemMessage(content=self.doctor_system_prompt.format(
                max_turns=self.max_turns,
                current_turn=1,
            )),
            HumanMessage(content=patient_opening),
        ]

        try:
            for turn_idx in range(self.max_turns):
                turn_count = turn_idx + 1
                is_last_turn = (turn_idx == self.max_turns - 1)

                # ── 每轮更新 system prompt 中的 current_turn ─────────────────
                # 替换 messages[0]，使模型始终知道当前是第几轮
                prompt_template = (
                    DOCTOR_FINAL_DIAGNOSIS_PROMPT
                    if is_last_turn and self.force_diagnose_on_last_turn
                    else self.doctor_system_prompt
                )
                messages[0] = SystemMessage(content=prompt_template.format(
                    max_turns=self.max_turns,
                    current_turn=turn_count,
                ))

                # ── Doctor 生成回复（传入完整对话历史）────────────────────────
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

                # Last-turn hard constraint: ask is not a valid final action.
                # We retry once with an explicit correction instead of rewriting
                # the action, so the recorded trajectory is still model-generated.
                if (
                    is_last_turn
                    and self.force_diagnose_on_last_turn
                    and action != "diagnose"
                ):
                    retry_messages = list(messages)
                    retry_messages.append(HumanMessage(
                        content=(
                            "[System] This is the final turn. Your previous output was not a diagnosis. "
                            "You must output exactly one JSON object with "
                            '{"action":"diagnose","diagnosis":"<best diagnosis>"}. '
                            "Do not ask another question."
                        )
                    ))
                    try:
                        retry_result = await model.ainvoke(
                            retry_messages,
                            sampling_params=sampling_params,
                        )
                        retry_response = retry_result.content
                        retry_action = self._parse_action(retry_response)
                        if retry_response is not None and retry_action == "diagnose":
                            result = retry_result
                            doctor_response = retry_response
                            action = retry_action
                    except MaxTokenExceededError:
                        break
                    except Exception as e:
                        print(f"[DiagPRMAgentLoop] Doctor final-diagnosis retry error: {e}")

                # ── 将 doctor 回复追加到对话历史 ─────────────────────────────
                # 必须保留 ChatModel 写入的 response_metadata，后续轮次会基于
                # 其中的 cumulative prompt_ids / response_mask 继续拼接。
                messages.append(result)

                # ── KG 工具处理（三种查询，可同时出现）──────────────────────
                if self._kg is not None:
                    kg_feedback_parts = []
                    parsed_json = self._parse_doctor_json(doctor_response)

                    # 1. query_kg: 疾病名 → KG 标准名（诊断时使用）
                    kg_query = parsed_json.get("query_kg", "").strip() if isinstance(parsed_json.get("query_kg"), str) else ""
                    if kg_query:
                        verbatim_name = lookup_disease_name(kg_query, self._kg)
                        if verbatim_name:
                            kg_feedback_parts.append(
                                f"[KG:disease_name] \"{kg_query}\" → official name: \"{verbatim_name}\". "
                                f"Use this exact string in your \"diagnosis\" field."
                            )
                        else:
                            kg_feedback_parts.append(
                                f"[KG:disease_name] \"{kg_query}\" → not found in KG."
                            )
                        print(f"[DiagPRMAgentLoop] KG query_kg: '{kg_query}' → '{verbatim_name}'")

                    # 2. query_kg_symptom: 症状 → 含该症状的疾病候选列表（初始假设形成）
                    kg_sym_queries = parsed_json.get("query_kg_symptom", [])
                    if isinstance(kg_sym_queries, str):
                        kg_sym_queries = [kg_sym_queries]
                    for sym_q in kg_sym_queries:
                        if not isinstance(sym_q, str) or not sym_q.strip():
                            continue
                        sym_q = sym_q.strip()
                        results = query_diseases_by_symptom(sym_q, self._kg, top_k=5)
                        if results:
                            diseases_str = ", ".join(f"\"{d}\" (w={w:.2f})" for d, w in results)
                            kg_feedback_parts.append(
                                f"[KG:symptom→diseases] symptom \"{sym_q}\" appears in: {diseases_str}."
                            )
                        else:
                            kg_feedback_parts.append(
                                f"[KG:symptom→diseases] symptom \"{sym_q}\" → no matching diseases found."
                            )
                        print(f"[DiagPRMAgentLoop] KG query_symptom: '{sym_q}' → {[d for d,_ in results]}")

                    # 3. query_kg_disease_symptoms: 疾病名 → 该病的 KG 症状列表（定向问诊）
                    kg_dis_queries = parsed_json.get("query_kg_disease_symptoms", [])
                    if isinstance(kg_dis_queries, str):
                        kg_dis_queries = [kg_dis_queries]
                    for dis_q in kg_dis_queries:
                        if not isinstance(dis_q, str) or not dis_q.strip():
                            continue
                        dis_q = dis_q.strip()
                        results = query_symptoms_by_disease(dis_q, self._kg, top_k=10)
                        if results:
                            syms_str = ", ".join(f"\"{s}\" (w={w:.2f})" for s, w in results)
                            kg_feedback_parts.append(
                                f"[KG:disease→symptoms] disease \"{dis_q}\" has symptoms: {syms_str}. "
                                f"Ask about unconfirmed symptoms to gather evidence."
                            )
                        else:
                            kg_feedback_parts.append(
                                f"[KG:disease→symptoms] disease \"{dis_q}\" → not found in KG."
                            )
                        print(f"[DiagPRMAgentLoop] KG query_disease_syms: '{dis_q}' → {[s for s,_ in results]}")

                    if kg_feedback_parts:
                        messages.append(HumanMessage(content="\n".join(kg_feedback_parts)))

                if action == "diagnose" or is_last_turn:
                    verifier_responses.append("<Normal>")
                    # 记录最终轮（诊断轮无 patient 回答）
                    dialogue_history.append({
                        "turn_id": turn_idx,
                        "doctor_response": doctor_response,
                        "patient_answer": "",
                        "patient_fact_id": "unknown",
                        "is_final": True,
                    })
                    break

                # ── 提取医生的问题 ─────────────────────────────────────────────
                question = self._parse_question(doctor_response)
                if question is None:
                    verifier_responses.append("<ERROR_RESPONSE>")
                    # 用系统提示告知医生格式有误，追加到历史
                    messages.append(HumanMessage(
                        content="[System] Your last response had no valid question. Please ask exactly one focused question."
                    ))
                    break

                # ── Verifier 检测（重复/多重问题）─────────────────────────────
                if self.verifier_enabled:
                    verifier_tag = self._run_verifier_sync(question, previous_questions)
                else:
                    verifier_tag = "<Normal>"
                verifier_responses.append(verifier_tag)

                if verifier_tag == "<Repeated>":
                    feedback = "You already asked this question. Please ask a different, more targeted question."
                    messages.append(HumanMessage(content=f"[System] {feedback}"))
                    previous_questions.append(question)
                    dialogue_history.append({
                        "turn_id": turn_idx,
                        "doctor_response": doctor_response,
                        "patient_answer": "",
                        "patient_fact_id": "unknown",
                    })
                    continue
                elif verifier_tag == "<Multiple>":
                    feedback = "You asked multiple questions at once. Please ask exactly one focused question."
                    messages.append(HumanMessage(content=f"[System] {feedback}"))
                    previous_questions.append(question)
                    dialogue_history.append({
                        "turn_id": turn_idx,
                        "doctor_response": doctor_response,
                        "patient_answer": "",
                        "patient_fact_id": "unknown",
                    })
                    continue

                # ── Patient Simulator 回答（LLM API）──────────────────────────
                patient_raw = await self._llm_patient(symptom_facts, question)

                # 解析 patient 回复中的 answer 和 hidden fact_id 字段
                patient_answer = self._parse_patient_answer(patient_raw)
                patient_fact_id = self._parse_patient_fact_id(
                    patient_raw,
                    fact_id_to_text=fact_id_to_text,
                    fact_text_to_id=fact_text_to_id,
                )
                patient_fact_text = fact_id_to_text.get(patient_fact_id, "")

                # ── 将 patient 回答追加到对话历史 ────────────────────────────
                # Doctor 只看到自然语言 answer；hidden fact_id/text 只进入
                # statistics，供 Reward Manager 和 rollout debug 使用。
                messages.append(HumanMessage(content=patient_answer))

                # ── 更新累积症状（供 reward 计算用）──────────────────────────
                if patient_fact_text:
                    if patient_fact_text not in collected_symptoms:
                        collected_symptoms.append(patient_fact_text)

                previous_questions.append(question)

                # 记录本轮对话（供 Reward Manager 使用）
                dialogue_history.append({
                    "turn_id": turn_idx,
                    "doctor_response": doctor_response,
                    "patient_answer": patient_answer,
                    "patient_fact_id": patient_fact_id,  # hidden fact id or "unknown"
                    "patient_fact_text": patient_fact_text,  # hidden debug/resolution text
                })

        except MaxTokenExceededError:
            pass
        except Exception as e:
            print(f"[DiagPRMAgentLoop] Unexpected error: {e}")
            import traceback
            traceback.print_exc()

        # ── 从最后的 AI 消息中提取完整 multi-turn token 序列 ────────────────
        # ChatModel 会在每轮 AIMessage.response_metadata 中累积：
        #   prompt_ids     = 初始 prompt + 所有 assistant/human/KG feedback tokens
        #   response_mask  = 初始 prompt 之后各 token 的 mask（assistant=1, human/KG=0）
        # 因此按照 ATPO 传统 multi-turn 格式切分：
        #   response_ids = prompt_ids[-len(response_mask):]
        #   prompt_ids   = prompt_ids[:-len(response_mask)]
        if last_ai_msg is None or "prompt_ids" not in last_ai_msg.response_metadata:
            # 兜底：用 tokenizer 编码最后一轮的 [system, user] 消息
            try:
                all_ids = self.tokenizer.apply_chat_template(
                    [{"role": "system", "content": self.doctor_system_prompt.format(
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

        prompt_ids = list(prompt_ids)
        response_mask = list(response_mask)

        if response_mask:
            pure_response_ids = prompt_ids[-len(response_mask):]
            pure_prompt_ids = prompt_ids[:-len(response_mask)]
            pure_response_mask = response_mask
        else:
            pure_prompt_ids = prompt_ids
            pure_response_ids = []
            pure_response_mask = []

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
                # 完整对话轨迹：供 Reward Manager 直接计算 turn-level reward。
                # Doctor 可见字段只有 doctor_response / patient_answer；
                # patient_fact_id 和 fact_id_to_text 是 hidden oracle signal。
                "dialogue_history": dialogue_history,
                "fact_id_to_text": fact_id_to_text,
                # _update_statistics 需要这三个键（可以为空列表）
                "q_value_variance_list": [],
                "mdp_value_list": [],
                "critic_value_list": [],
            },
        )
        return [output]

    async def _llm_patient(
        self,
        symptom_facts: List[Dict[str, Any]],
        question: str,
    ) -> str:
        """
        LLM-based patient simulator.

        Calls an external OpenAI-compatible chat API with PATIENT_SYSTEM_PROMPT
        (containing the atomic facts) and the doctor's question as the user message.
        Falls back to a simple rule-based answer if the API call fails.
        """
        # Format the hidden fact sheet as id: symptom. The Patient returns only
        # fact_id, while the Doctor receives only the natural-language answer.
        facts_text = "\n".join(
            f"- {f['fact_id']}: {f['text']}" for f in symptom_facts
        ) if symptom_facts else "(no known facts)"
        system_content = PATIENT_SYSTEM_PROMPT.format(atomic_facts=facts_text)

        payload = {
            "model": self.patient_model,
            "messages": [
                {"role": "system", "content": system_content},
                {"role": "user", "content": question},
            ],
            "max_tokens": self.patient_max_tokens,
            "temperature": 0.0,
            "stream": False,
        }
        # enable_thinking 是美团内部 API 特有字段（qwen3.5-plus 等思考模型）
        # 本地 vLLM server 不支持此字段，只对外网 API 发送
        _is_local = self.patient_api_base.startswith("http://127.0.0.1") or \
                    self.patient_api_base.startswith("http://localhost")
        if not self.patient_enable_thinking and not _is_local:
            payload["enable_thinking"] = False
        headers = {"Content-Type": "application/json"}
        api_key = os.environ.get("PATIENT_API_KEY", "")
        if api_key and not _is_local:
            headers["Authorization"] = f"Bearer {api_key}"

        url = self.patient_api_base.rstrip("/") + "/chat/completions"
        # 本地 vLLM 无需代理；外网 API 读取代理配置
        _proxy = None if _is_local else (
            os.environ.get("PATIENT_PROXY") or os.environ.get("https_proxy") or os.environ.get("HTTPS_PROXY")
        )
        _sem = _get_patient_sem() if not _is_local else None
        for _attempt in range(8):
            try:
                async with (_sem if _sem else nullcontext()):
                    async with aiohttp.ClientSession() as session:
                        async with session.post(
                            url,
                            json=payload,
                            headers=headers,
                            timeout=aiohttp.ClientTimeout(total=60),
                            proxy=_proxy,
                        ) as resp:
                            if resp.status == 200:
                                data = await resp.json()
                                _msg = data["choices"][0]["message"]
                                _raw = _strip_thinking_text(_msg.get("content") or "")
                                return _raw
                            elif resp.status == 429:
                                wait = min(2 ** _attempt, 60)
                                text = await resp.text()
                                print(f"[Patient API] 429 rate limit, retry {_attempt+1}/8 after {wait}s")
                                await asyncio.sleep(wait)
                            else:
                                text = await resp.text()
                                print(f"[Patient API] HTTP {resp.status}: {text[:200]}")
                                break
            except Exception as e:
                print(f"[Patient API] call failed: {e}")
                if _attempt < 7:
                    await asyncio.sleep(min(2 ** _attempt, 60))

        # Fallback: simple keyword match against hidden symptom facts.
        return self._rule_based_patient_fallback(symptom_facts, question)

    def _rule_based_patient_fallback(
        self,
        symptom_facts: List[Dict[str, Any]],
        question: str,
    ) -> str:
        """Fallback rule-based patient answer used when the LLM API is unavailable."""
        question_lower = question.lower()
        keywords = set(re.findall(r'\b[a-z]{4,}\b', question_lower))
        keywords -= {"have", "does", "your", "you", "the", "any", "this", "that",
                     "please", "tell", "about", "been", "feel", "experiencing",
                     "patient", "currently", "present", "symptom", "condition"}
        if symptom_facts and keywords:
            for fact in symptom_facts:
                fact_text = fact["text"]
                if any(kw in fact_text.lower() for kw in keywords):
                    return json.dumps(
                        {
                            "answer": f"Yes, I have been experiencing {fact_text}.",
                            "fact_id": fact["fact_id"],
                        },
                        ensure_ascii=False,
                    )
        return json.dumps(
            {"answer": "I'm not sure about that.", "fact_id": "unknown"},
            ensure_ascii=False,
        )

    def _build_symptom_facts(
        self,
        symptoms_pool: Any,
        initial_symptoms: List[str],
        ground_truth: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Build a stable hidden fact sheet [{fact_id, text, weight}, ...]."""
        raw_facts = ground_truth.get("symptom_facts", None)
        if isinstance(raw_facts, str):
            try:
                raw_facts = json.loads(raw_facts)
            except Exception:
                raw_facts = None
        if isinstance(raw_facts, list) and raw_facts:
            facts = []
            for idx, item in enumerate(raw_facts):
                if isinstance(item, dict):
                    text = str(item.get("text", "")).strip()
                    if not text:
                        continue
                    facts.append({
                        "fact_id": str(item.get("fact_id") or f"F{idx:03d}"),
                        "text": text,
                        "weight": float(item.get("weight", 1.0)),
                    })
            if facts:
                return facts

        items: List[Tuple[str, float]] = []
        if isinstance(symptoms_pool, str):
            try:
                symptoms_pool = json.loads(symptoms_pool)
            except Exception:
                symptoms_pool = {}
        if isinstance(symptoms_pool, dict) and symptoms_pool:
            items = [(str(sym), float(weight)) for sym, weight in symptoms_pool.items()]
        else:
            legacy = ground_truth.get("atomic_facts", None)
            if isinstance(legacy, str):
                try:
                    legacy = json.loads(legacy)
                except Exception:
                    legacy = [legacy]
            if isinstance(legacy, list) and legacy:
                items = [(str(sym), 1.0) for sym in legacy]
            else:
                items = [(str(sym), 1.0) for sym in (initial_symptoms or [])]

        facts = []
        seen = set()
        for text, weight in items:
            text = text.strip()
            if not text or text in seen:
                continue
            seen.add(text)
            facts.append({
                "fact_id": f"F{len(facts):03d}",
                "text": text,
                "weight": float(weight),
            })
        return facts

    async def _llm_patient_opening(
        self,
        initial_symptoms: List[str],
    ) -> str:
        """
        让 Patient LLM 根据 initial_symptoms 生成自然语言开场白（仅主诉）。

        只告知患者"这是你现在需要描述的症状"，严格禁止透露诊断。
        失败时回退到简单拼接：
          "Hello doctor, I've been experiencing {sym1}, {sym2}."

        Returns:
            患者的自然语言开场白字符串（纯文本，无 JSON）。
        """
        if not initial_symptoms:
            return "Hello doctor, I haven't been feeling well lately and would like to get checked out."

        syms_str = "\n".join(f"- {s}" for s in initial_symptoms)
        system_content = PATIENT_OPENING_PROMPT.format(initial_symptoms=syms_str)

        payload = {
            "model": self.patient_model,
            "messages": [
                {"role": "system", "content": system_content},
                {"role": "user", "content": "Please introduce yourself to the doctor now."},
            ],
            "max_tokens": self.patient_max_tokens,
            "temperature": 0.7,
            "stream": False,
        }
        _is_local = self.patient_api_base.startswith("http://127.0.0.1") or \
                    self.patient_api_base.startswith("http://localhost")
        if not self.patient_enable_thinking and not _is_local:
            payload["enable_thinking"] = False
        headers = {"Content-Type": "application/json"}
        api_key = os.environ.get("PATIENT_API_KEY", "")
        if api_key and not _is_local:
            headers["Authorization"] = f"Bearer {api_key}"

        url = self.patient_api_base.rstrip("/") + "/chat/completions"
        _proxy = None if _is_local else (
            os.environ.get("PATIENT_PROXY") or os.environ.get("https_proxy") or os.environ.get("HTTPS_PROXY")
        )
        _sem = _get_patient_sem() if not _is_local else None
        for _attempt in range(8):
            try:
                async with (_sem if _sem else nullcontext()):
                    async with aiohttp.ClientSession() as session:
                        async with session.post(
                            url,
                            json=payload,
                            headers=headers,
                            timeout=aiohttp.ClientTimeout(total=60),
                            proxy=_proxy,
                        ) as resp:
                            if resp.status == 200:
                                data = await resp.json()
                                _msg = data["choices"][0]["message"]
                                opening = _strip_thinking_text(_msg.get("content") or "")
                                if opening:
                                    return opening
                            elif resp.status == 429:
                                wait = min(2 ** _attempt, 60)
                                text = await resp.text()
                                print(f"[Patient Opening API] 429 rate limit, retry {_attempt+1}/8 after {wait}s")
                                await asyncio.sleep(wait)
                            else:
                                text = await resp.text()
                                print(f"[Patient Opening API] HTTP {resp.status}: {text[:200]}")
                                break
            except Exception as e:
                print(f"[Patient Opening API] call failed: {e}")
                if _attempt < 7:
                    await asyncio.sleep(min(2 ** _attempt, 60))

        # Fallback: 直接拼接
        syms_natural = ", ".join(initial_symptoms)
        return f"Hello doctor, I've been experiencing {syms_natural}. I'd like to get checked out."

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

    def _parse_doctor_json(self, doctor_response: str) -> Dict:
        """从 doctor 回复中解析 JSON 块。

        支持三种格式：
          1. ```json ... ``` 代码块
          2. 裸 JSON 对象 {...}
          3. 模型内部 <think> 后的 JSON

        Returns:
            解析后的 dict，失败时返回 {}
        """
        # 1. 尝试提取 ```json ... ``` 代码块
        code_block = re.search(r'```json\s*([\s\S]*?)```', doctor_response, re.IGNORECASE)
        if code_block:
            try:
                result = json.loads(code_block.group(1).strip())
                if isinstance(result, dict):
                    return result
            except Exception:
                pass

        # 2. 去掉 <think>...</think> 再找 {...}
        stripped = re.sub(r'<think>[\s\S]*?</think>', '', doctor_response, flags=re.IGNORECASE)
        brace_match = re.search(r'(\{[\s\S]*\})', stripped)
        if brace_match:
            try:
                result = json.loads(brace_match.group(1).strip())
                if isinstance(result, dict):
                    return result
            except Exception:
                pass

        # 3. 对整个 response 尝试（兜底）
        brace_match2 = re.search(r'(\{[\s\S]*\})', doctor_response)
        if brace_match2:
            try:
                result = json.loads(brace_match2.group(1).strip())
                if isinstance(result, dict):
                    return result
            except Exception:
                pass

        return {}

    def _parse_hypothesis_name(self, doctor_response: str) -> Tuple[Optional[str], Optional[str]]:
        """Backward-compatible parser for old hypothesis-format outputs.

        Returns:
            (hypothesis_name | None, action | None)
        """
        parsed = self._parse_doctor_json(doctor_response)
        hypothesis = parsed.get("hypothesis") or None
        if hypothesis:
            hypothesis = str(hypothesis).strip()
        raw_action = self._normalize_action_value(parsed.get("action", ""))
        if raw_action in ("ask", "continue", "switch"):
            action = "ask"
        elif raw_action == "diagnose":
            action = "diagnose"
        else:
            action = None
        return hypothesis, action

    def _parse_patient_json(self, raw: str) -> Dict:
        """从 patient 回复中解析 JSON 块。

        Returns:
            解析后的 dict，失败时返回 {}
        """
        # 1. 尝试提取 ```json ... ``` 代码块
        code_block = re.search(r'```json\s*([\s\S]*?)```', raw, re.IGNORECASE)
        if code_block:
            try:
                result = json.loads(code_block.group(1).strip())
                if isinstance(result, dict):
                    return result
            except Exception:
                pass

        # 2. 找 {...} 对象
        brace_match = re.search(r'(\{[\s\S]*\})', raw)
        if brace_match:
            try:
                result = json.loads(brace_match.group(1).strip())
                if isinstance(result, dict):
                    return result
            except Exception:
                pass

        return {}

    def _parse_patient_answer(self, raw: str) -> str:
        """Extract the answer field from patient LLM output (JSON format)."""
        parsed = self._parse_patient_json(raw)
        if parsed.get("answer"):
            return str(parsed["answer"]).strip()
        # Fallback: legacy XML tag
        match = re.search(r'<answer>\s*(.*?)\s*</answer>', raw, re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1).strip()
        return raw.strip()

    def _parse_patient_fact(self, raw: str) -> str:
        """Extract the fact field from patient LLM output (JSON format).

        Returns the atomic fact string, or 'unknown' if not present / explicitly unknown.
        """
        parsed = self._parse_patient_json(raw)
        if "fact" in parsed:
            fact = str(parsed["fact"]).strip()
            return fact if fact and fact.lower() != "unknown" else "unknown"
        # Fallback: legacy XML tag
        match = re.search(r'<fact>\s*(.*?)\s*</fact>', raw, re.IGNORECASE | re.DOTALL)
        if match:
            fact = match.group(1).strip()
            return fact if fact else "unknown"
        return "unknown"

    def _parse_patient_fact_id(
        self,
        raw: str,
        fact_id_to_text: Dict[str, str],
        fact_text_to_id: Dict[str, str],
    ) -> str:
        """Extract hidden fact_id from patient output, with legacy fact fallback."""
        parsed = self._parse_patient_json(raw)
        if "fact_id" in parsed:
            fact_id = str(parsed["fact_id"]).strip()
            if not fact_id or fact_id.lower() == "unknown":
                return "unknown"
            return fact_id if fact_id in fact_id_to_text else "unknown"

        # Backward compatibility for old patient prompts that returned verbatim fact.
        legacy_fact = self._parse_patient_fact(raw)
        if legacy_fact and legacy_fact.lower() != "unknown":
            if legacy_fact in fact_text_to_id:
                return fact_text_to_id[legacy_fact]
            norm_legacy = legacy_fact.strip().lower()
            for text, fact_id in fact_text_to_id.items():
                if text.strip().lower() == norm_legacy:
                    return fact_id
        return "unknown"

    def _is_negative_answer(self, answer: str) -> bool:
        """Return True if the patient's answer is a denial (no new symptom confirmed)."""
        deny_pattern = re.compile(
            r'^\s*(no|not|i don\'t|i\'m not|i do not|i have not|i\'m not sure|i don\'t have|i do not have)',
            re.IGNORECASE,
        )
        return bool(deny_pattern.match(answer.strip()))

    def _parse_action(self, doctor_response: str) -> Optional[str]:
        """从 doctor 回复中解析 action（JSON 格式）。
        动作空间： ask / diagnose（将旧的 continue/switch 平滑合并为 ask）
        """
        parsed = self._parse_doctor_json(doctor_response)
        raw = self._normalize_action_value(parsed.get("action", ""))
        if raw in ("ask", "continue", "switch"):
            return "ask"
        if raw == "diagnose":
            return "diagnose"
        # Fallback: legacy XML tag
        match = re.search(
            r'<action>\s*(ask|continue|switch|diagnose)\s*</action>',
            doctor_response,
            re.IGNORECASE | re.DOTALL,
        )
        if match:
            raw2 = match.group(1).strip().lower()
            return "ask" if raw2 in ("ask", "continue", "switch") else raw2
        return None

    def _normalize_action_value(self, value: Any) -> str:
        """Normalize potentially malformed doctor action values.

        During RL, the actor may occasionally emit malformed JSON such as
        {"action": {"type": "ask"}}. Treat these as parseable when possible
        and invalid otherwise, instead of crashing the rollout worker.
        """
        if value is None:
            return ""
        if isinstance(value, str):
            return value.strip().lower()
        if isinstance(value, dict):
            for key in ("action", "type", "name", "value"):
                nested = self._normalize_action_value(value.get(key))
                if nested:
                    return nested
            return ""
        if isinstance(value, list):
            for item in value:
                nested = self._normalize_action_value(item)
                if nested:
                    return nested
            return ""
        return str(value).strip().lower()

    def _parse_question(self, doctor_response: str) -> Optional[str]:
        """从 doctor 回复中解析 question 字段（JSON 格式）。"""
        parsed = self._parse_doctor_json(doctor_response)
        if parsed.get("question"):
            return str(parsed["question"]).strip()
        # Fallback: legacy XML tag
        match = re.search(
            r'<question>\s*(.*?)\s*</question>',
            doctor_response,
            re.IGNORECASE | re.DOTALL,
        )
        if match:
            return match.group(1).strip()
        return None

    def _parse_kg_query(self, doctor_response: str) -> Optional[str]:
        """从 doctor 回复中解析 'query_kg' 字段（JSON 格式）。
        
        Returns:
            str  — 医生想查询的疾病名（原始字符串，未经 normalize）
            None — 本轮没有 query_kg 字段
        """
        parsed = self._parse_doctor_json(doctor_response)
        query = parsed.get("query_kg")
        if query and isinstance(query, str):
            return query.strip()
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
        """构建初始消息列表（兜底路径）。"""
        system_msg = SystemMessage(content=self.doctor_system_prompt.format(
            max_turns=self.max_turns,
            current_turn=1,
        ))
        user_msg = HumanMessage(content=chief_complaint)
        return [system_msg, user_msg]
