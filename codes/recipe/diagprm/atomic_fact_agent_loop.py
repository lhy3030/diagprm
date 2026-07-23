"""ATPO-style atomic-fact medical QA agent loop.

The actor uses the ATPO action format:
  Question: ...
  Final Answer: A

The environment keeps hidden atomic facts and returns natural patient/case
answers plus a hidden fact_id for reward computation.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from contextlib import nullcontext
from typing import Any

import aiohttp
from langchain_core.messages import AIMessage, HumanMessage, convert_to_messages

from recipe.atpo.agent_loop import AgentLoopBase, AgentLoopMetrics, AgentLoopOutput
from recipe.atpo.chat_model import ChatModel, MaxTokenExceededError
from recipe.diagprm.prompts.atomic_fact import (
    ATOMIC_FACT_DOCTOR_SYSTEM_PROMPT,
    ATOMIC_FACT_PATIENT_SYSTEM_PROMPT,
    ATOMIC_FACT_SFT_MAX_TURNS,
)


_PATIENT_SEM = None
_CURRENT_TURN_RE = re.compile(
    r"\n+Current turn:\s*\d+\s*/\s*\d+\.\s*$",
    flags=re.IGNORECASE,
)


def _get_patient_sem():
    global _PATIENT_SEM
    if _PATIENT_SEM is None:
        limit = int(os.environ.get("PATIENT_API_CONCURRENCY", "16"))
        _PATIENT_SEM = asyncio.Semaphore(limit)
    return _PATIENT_SEM


def _strip_thinking_text(text: str) -> str:
    text = text or ""
    text = re.sub(r"<think>[\s\S]*?</think>", "", text, flags=re.IGNORECASE)
    return text.strip()


def _as_dict(value: Any) -> dict:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            obj = json.loads(value)
            return obj if isinstance(obj, dict) else {}
        except Exception:
            return {}
    if hasattr(value, "item"):
        try:
            obj = value.item()
            return obj if isinstance(obj, dict) else {}
        except Exception:
            return {}
    try:
        return dict(value) if value is not None else {}
    except Exception:
        return {}


def _as_list(value: Any) -> list:
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            obj = json.loads(value)
            return obj if isinstance(obj, list) else [value]
        except Exception:
            return [value]
    if value is None:
        return []
    try:
        return list(value)
    except Exception:
        return [value]


def _trailing_generated_token_ids(message: AIMessage) -> list[int]:
    """Return token ids for the most recent assistant generation."""
    metadata = getattr(message, "response_metadata", {}) or {}
    token_ids = list(metadata.get("prompt_ids", []) or [])
    response_mask = list(metadata.get("response_mask", []) or [])
    if not token_ids or not response_mask:
        return []
    n = 0
    for value in reversed(response_mask):
        if int(value) != 1:
            break
        n += 1
    if n <= 0:
        return []
    return token_ids[-n:]


def _to_openai_messages(messages: list[Any]) -> list[dict[str, str]]:
    role_map = {"system": "system", "human": "user", "ai": "assistant"}
    out = []
    for message in messages:
        role = role_map.get(getattr(message, "type", ""), getattr(message, "type", "user"))
        out.append({"role": role, "content": str(getattr(message, "content", ""))})
    return out


def _set_current_turn_on_latest_user(
    messages: list[Any], current_turn: int, max_turns: int
) -> None:
    for message in reversed(messages):
        if getattr(message, "type", "") != "human":
            continue
        content = _CURRENT_TURN_RE.sub("", str(message.content).rstrip())
        message.content = f"{content}\n\nCurrent turn: {current_turn} / {max_turns}."
        return
    raise ValueError("AtomicFactAgentLoop requires a user message for the current turn.")


class AtomicFactAgentLoop(AgentLoopBase):
    def __init__(self, trainer_config, server_manager, tokenizer, processor, **kwargs):
        super().__init__(trainer_config, server_manager, tokenizer, processor, **kwargs)
        self.max_turns = int(os.environ.get("MAX_TURNS_OVERRIDE") or kwargs.get("max_turns", 10))
        self.max_turn_response_tokens = int(
            os.environ.get("ATOMIC_FACT_MAX_TURN_TOKENS")
            or kwargs.get("max_turn_response_tokens", 1024)
        )
        self.min_turn_response_tokens = int(
            os.environ.get("ATOMIC_FACT_MIN_TURN_TOKENS")
            or kwargs.get("min_turn_response_tokens", 1)
        )
        self.verifier_enabled = bool(kwargs.get("verifier_enabled", True))
        self.patient_api_base = (
            os.environ.get("PATIENT_API_BASE")
            or kwargs.get("patient_api_base", "http://127.0.0.1:8100/v1")
        )
        self.patient_model = os.environ.get("PATIENT_MODEL") or kwargs.get("patient_model", "Qwen3-8B")
        self.patient_max_tokens = int(
            os.environ.get("PATIENT_MAX_TOKENS") or kwargs.get("patient_max_tokens", 512)
        )
        self.patient_enable_thinking = bool(kwargs.get("patient_enable_thinking", False))
        self.prompt_style = os.environ.get("ATOMIC_FACT_PROMPT_STYLE") or kwargs.get("prompt_style", "dataset")
        if self.prompt_style != "dataset":
            raise ValueError("AtomicFactAgentLoop only supports prompt_style='dataset' for ATPO-aligned runs.")
        self.doctor_enable_thinking = str(
            os.environ.get("ATOMIC_FACT_DOCTOR_THINKING", kwargs.get("doctor_enable_thinking", "0"))
        ).lower() not in {"0", "false", "no"}
        if self.max_turns != ATOMIC_FACT_SFT_MAX_TURNS:
            raise ValueError(
                f"AtomicFact RL must use max_turns={ATOMIC_FACT_SFT_MAX_TURNS} "
                "to match the SFT prompt."
            )
        if self.doctor_enable_thinking:
            raise ValueError("AtomicFact RL must use doctor_enable_thinking=False to match SFT.")
        self._wait_for_patient_vllm()

    def _wait_for_patient_vllm(self) -> None:
        """轮询等待 patient vLLM 就绪，超时前每 5s 检查一次。
        每次轮询都尝试从 PATIENT_VLLM_ENDPOINT_FILE 读取最新 endpoint，
        支持监控脚本在 RL 启动后动态写入真实 IP。
        """
        import time
        import urllib.request

        endpoint_file = os.environ.get(
            "PATIENT_VLLM_ENDPOINT_FILE",
            os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
                "patient_vllm.endpoint",
            ),
        )
        max_wait = int(os.environ.get("PATIENT_VLLM_MAX_WAIT", "1800"))
        interval = 5
        waited = 0
        print(
            f"[AtomicFactAgentLoop] Waiting for patient vLLM (max {max_wait}s). "
            f"Will check endpoint file: {endpoint_file}",
            flush=True,
        )
        while True:
            # 每次都尝试从文件读取最新 endpoint
            if os.path.exists(endpoint_file):
                try:
                    new_base = open(endpoint_file).read().strip()
                    if new_base and new_base != self.patient_api_base:
                        print(
                            f"[AtomicFactAgentLoop] Updated patient_api_base from file: {new_base}",
                            flush=True,
                        )
                        self.patient_api_base = new_base
                except Exception:
                    pass

            health_url = self.patient_api_base.rstrip("/").rstrip("v1").rstrip("/") + "/health"
            try:
                with urllib.request.urlopen(health_url, timeout=5) as resp:
                    if resp.status == 200:
                        print(
                            f"[AtomicFactAgentLoop] Patient vLLM ready after {waited}s at {self.patient_api_base}.",
                            flush=True,
                        )
                        return
            except Exception:
                pass

            if waited >= max_wait:
                raise TimeoutError(
                    f"[AtomicFactAgentLoop] Patient vLLM not ready after {max_wait}s at {self.patient_api_base}"
                )
            if waited % 30 == 0:
                print(
                    f"[AtomicFactAgentLoop] Still waiting for patient vLLM at {health_url} ({waited}s) ...",
                    flush=True,
                )
            time.sleep(interval)
            waited += interval

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> list[AgentLoopOutput]:
        rollout = self.config.actor_rollout_ref.rollout
        model_type = self.config.actor_rollout_ref.model_type
        model = ChatModel(
            model=model_type,
            client=self.server_manager,
            tokenizer=self.tokenizer,
            max_tokens=rollout.response_length,
            enable_thinking=self.doctor_enable_thinking,
        )

        extra_info = _as_dict(kwargs.get("extra_info", {}))
        reward_model = _as_dict(kwargs.get("reward_model", {}))
        ground_truth = _as_dict(reward_model.get("ground_truth", {}))
        answer_idx = str(
            kwargs.get("answer_idx")
            or extra_info.get("answer_idx")
            or ground_truth.get("answer")
            or ""
        ).strip().upper()

        raw_prompt = kwargs.get("raw_prompt") or kwargs.get("prompt") or extra_info.get("prompt", [])
        if isinstance(raw_prompt, str):
            try:
                raw_prompt = json.loads(raw_prompt)
            except Exception:
                raw_prompt = []
        if not raw_prompt:
            raise ValueError(
                "AtomicFactAgentLoop requires dataset prompt in dataset mode. "
                "Set data.return_raw_chat=True or store the prompt in extra_info.prompt."
            )
        messages = convert_to_messages(raw_prompt)
        if not messages or messages[0].type != "system":
            raise ValueError("AtomicFactAgentLoop requires a dataset prompt whose first message is system.")
        expected_system_prompt = ATOMIC_FACT_DOCTOR_SYSTEM_PROMPT.format(
            max_turns=self.max_turns
        )
        if str(messages[0].content) != expected_system_prompt:
            raise ValueError(
                "AtomicFact RL dataset system prompt does not match SFT. Run "
                "rewrite_atomic_fact_rl_prompts.py before training."
            )

        fact_items = self._build_fact_items(kwargs, extra_info)
        fact_id_to_text = {f["fact_id"]: f["text"] for f in fact_items}
        previous_questions: list[str] = []
        dialogue_history: list[dict[str, Any]] = []
        verifier_responses: list[str] = []
        last_ai_msg = None
        turn_count = 0

        try:
            for turn_idx in range(self.max_turns):
                turn_count = turn_idx + 1
                is_last_turn = turn_idx == self.max_turns - 1
                _set_current_turn_on_latest_user(messages, turn_count, self.max_turns)
                try:
                    turn_sampling_params = dict(sampling_params or {})
                    turn_sampling_params.setdefault("max_tokens", self.max_turn_response_tokens)
                    if self.min_turn_response_tokens > 0:
                        turn_sampling_params.setdefault("min_tokens", self.min_turn_response_tokens)
                    result = await model.ainvoke(messages, sampling_params=turn_sampling_params)
                except MaxTokenExceededError:
                    break
                except Exception as e:
                    print(f"[AtomicFactAgentLoop] Doctor call error: {e}")
                    break
                if result is None or result.content is None:
                    break

                doctor_response = result.content
                generated_token_ids = _trailing_generated_token_ids(result)
                eos_ids = {
                    x
                    for x in [
                        self.tokenizer.eos_token_id,
                        self.tokenizer.pad_token_id,
                    ]
                    if x is not None
                }
                hit_no_eos_limit = (
                    len(generated_token_ids) >= self.max_turn_response_tokens
                    and (not generated_token_ids or generated_token_ids[-1] not in eos_ids)
                )
                response_type = self._parse_response_type(doctor_response)

                messages.append(result)
                last_ai_msg = result

                if hit_no_eos_limit:
                    dialogue_history.append({
                        "turn_id": turn_idx,
                        "doctor_response": doctor_response,
                        "patient_answer": "",
                        "patient_fact_id": "unknown",
                        "action": "no_eos",
                        "is_final": True,
                        "is_no_eos": True,
                        "generated_tokens": len(generated_token_ids),
                    })
                    verifier_responses.append("<NO_EOS>")
                    break

                if response_type == "final_answer":
                    dialogue_history.append({
                        "turn_id": turn_idx,
                        "doctor_response": doctor_response,
                        "patient_answer": "",
                        "patient_fact_id": "unknown",
                        "action": "answer",
                        "is_final": True,
                    })
                    verifier_responses.append("<Normal>")
                    break

                question = self._parse_question(doctor_response)
                if not question:
                    messages.append(HumanMessage(content="[System] Your response must start with `Question:` or `Final Answer:`."))
                    dialogue_history.append({
                        "turn_id": turn_idx,
                        "doctor_response": doctor_response,
                        "patient_answer": "",
                        "patient_fact_id": "unknown",
                        "action": "invalid",
                    })
                    verifier_responses.append("<ERROR_RESPONSE>")
                    break

                verifier = self._verify_question(question, previous_questions)
                verifier_responses.append(verifier)
                if verifier == "<Repeated>":
                    patient_answer = "You already asked that. Please ask a different question."
                    fact_id = "unknown"
                elif verifier == "<Multiple>":
                    patient_answer = "Please ask one question at a time."
                    fact_id = "unknown"
                else:
                    patient_raw = await self._llm_patient(fact_items, question)
                    patient_answer = self._parse_patient_answer(patient_raw)
                    fact_id = self._parse_patient_fact_id(patient_raw, fact_id_to_text)

                messages.append(HumanMessage(content=patient_answer))
                previous_questions.append(question)
                dialogue_history.append({
                    "turn_id": turn_idx,
                    "doctor_response": doctor_response,
                    "patient_answer": patient_answer,
                    "patient_fact_id": fact_id,
                    "patient_fact_text": fact_id_to_text.get(fact_id, ""),
                    "action": "ask",
                })

        except Exception as e:
            print(f"[AtomicFactAgentLoop] Unexpected error: {e}")
            import traceback
            traceback.print_exc()

        prompt_ids, response_ids, response_mask = self._extract_token_sequences(
            last_ai_msg, messages, turn_count, rollout
        )
        return [AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids,
            response_mask=response_mask,
            num_turns=turn_count,
            verifier_responses=verifier_responses or ["<Normal>"],
            metrics=AgentLoopMetrics(),
            statistics={
                "answer_idx": answer_idx,
                "doctor_system_prompt": ATOMIC_FACT_DOCTOR_SYSTEM_PROMPT.format(
                    max_turns=self.max_turns
                ),
                "doctor_enable_thinking": self.doctor_enable_thinking,
                "max_turns": self.max_turns,
                "dialogue_history": dialogue_history,
                "fact_id_to_text": fact_id_to_text,
                "q_value_variance_list": [],
                "mdp_value_list": [],
                "critic_value_list": [],
            },
        )]

    def _build_fact_items(self, kwargs: dict, extra_info: dict) -> list[dict[str, Any]]:
        raw_items = extra_info.get("atomic_fact_items") or kwargs.get("atomic_fact_items")
        raw_items = _as_list(raw_items)
        facts: list[dict[str, Any]] = []
        for idx, item in enumerate(raw_items):
            item = _as_dict(item)
            text = str(item.get("text", "")).strip()
            if text:
                facts.append({
                    "fact_id": str(item.get("fact_id") or f"F{idx:03d}"),
                    "text": text,
                    "weight": float(item.get("weight", 1.0)),
                })
        if facts:
            return facts
        raw_facts = (
            kwargs.get("atomic_facts")
            or extra_info.get("atomic_facts")
            or _as_dict(kwargs.get("reward_model", {})).get("ground_truth", {}).get("atomic_facts", [])
        )
        out = []
        seen = set()
        for fact in _as_list(raw_facts):
            text = str(fact).strip()
            if not text or text.lower() in seen:
                continue
            seen.add(text.lower())
            out.append({"fact_id": f"F{len(out):03d}", "text": text, "weight": 1.0})
        return out

    async def _llm_patient(self, fact_items: list[dict[str, Any]], question: str) -> str:
        facts_text = "\n".join(f"- {f['fact_id']}: {f['text']}" for f in fact_items) or "(no known facts)"
        payload = {
            "model": self.patient_model,
            "messages": [
                {"role": "system", "content": ATOMIC_FACT_PATIENT_SYSTEM_PROMPT.format(atomic_facts=facts_text)},
                {"role": "user", "content": question},
            ],
            "max_tokens": self.patient_max_tokens,
            "temperature": 0.0,
            "stream": False,
        }
        is_local = self.patient_api_base.startswith("http://127.0.0.1") or self.patient_api_base.startswith("http://localhost")
        if not self.patient_enable_thinking and not is_local:
            payload["enable_thinking"] = False
        headers = {"Content-Type": "application/json"}
        api_key = os.environ.get("PATIENT_API_KEY", "")
        if api_key and not is_local:
            headers["Authorization"] = f"Bearer {api_key}"
        url = self.patient_api_base.rstrip("/") + "/chat/completions"
        proxy = None if is_local else (
            os.environ.get("PATIENT_PROXY") or os.environ.get("https_proxy") or os.environ.get("HTTPS_PROXY")
        )
        sem = None if is_local else _get_patient_sem()
        for attempt in range(5):
            try:
                async with (sem if sem else nullcontext()):
                    async with aiohttp.ClientSession() as session:
                        async with session.post(
                            url,
                            json=payload,
                            headers=headers,
                            timeout=aiohttp.ClientTimeout(total=60),
                            proxy=proxy,
                        ) as resp:
                            if resp.status == 200:
                                data = await resp.json()
                                content = data["choices"][0]["message"].get("content", "")
                                return _strip_thinking_text(content)
                            if resp.status == 429:
                                await asyncio.sleep(min(2 ** attempt, 30))
                                continue
                            text = await resp.text()
                            print(f"[AtomicFact Patient API] HTTP {resp.status}: {text[:200]}")
                            break
            except Exception as e:
                print(f"[AtomicFact Patient API] call failed: {e}")
                await asyncio.sleep(min(2 ** attempt, 30))
        return self._rule_patient(fact_items, question)

    def _rule_patient(self, fact_items: list[dict[str, Any]], question: str) -> str:
        q_words = set(re.findall(r"\b[a-z]{4,}\b", question.lower()))
        q_words -= {"patient", "doctor", "symptom", "symptoms", "please", "about", "have", "with", "that", "this"}
        best = None
        best_score = 0
        for fact in fact_items:
            words = set(re.findall(r"\b[a-z]{4,}\b", fact["text"].lower()))
            score = len(q_words & words)
            if score > best_score:
                best = fact
                best_score = score
        if best and best_score > 0:
            return json.dumps({
                "answer": f"Yes. {best['text']}",
                "fact_id": best["fact_id"],
            })
        return json.dumps({"answer": "The patient cannot answer this question.", "fact_id": "unknown"})

    def _parse_patient_json(self, raw: str) -> dict:
        raw = _strip_thinking_text(raw)
        match = re.search(r"```json\s*([\s\S]*?)```", raw, re.IGNORECASE)
        if match:
            raw = match.group(1)
        else:
            match = re.search(r"(\{[\s\S]*\})", raw)
            raw = match.group(1) if match else raw
        try:
            obj = json.loads(raw.strip())
            return obj if isinstance(obj, dict) else {}
        except Exception:
            return {}

    def _parse_patient_answer(self, raw: str) -> str:
        obj = self._parse_patient_json(raw)
        if obj.get("answer"):
            return str(obj["answer"]).strip()
        return _strip_thinking_text(raw) or "The patient cannot answer this question."

    def _parse_patient_fact_id(self, raw: str, fact_id_to_text: dict[str, str]) -> str:
        obj = self._parse_patient_json(raw)
        fact_id = str(obj.get("fact_id", "unknown")).strip()
        if fact_id in fact_id_to_text:
            return fact_id
        return "unknown"

    def _parse_response_type(self, text: str) -> str:
        stripped = _strip_thinking_text(text)
        if re.search(r"^\s*Final Answer\s*:", stripped, re.IGNORECASE):
            return "final_answer"
        if re.search(r"^\s*Question\s*:", stripped, re.IGNORECASE):
            return "question"
        return "invalid"

    def _parse_question(self, text: str) -> str | None:
        stripped = _strip_thinking_text(text)
        match = re.search(r"^\s*Question\s*:\s*([\s\S]+?)\s*$", stripped, re.IGNORECASE)
        return match.group(1).strip() if match else None

    def _verify_question(self, question: str, previous: list[str]) -> str:
        if not self.verifier_enabled:
            return "<Normal>"
        if question.count("?") > 1:
            return "<Multiple>"
        q_tokens = set(question.lower().split())
        for prev in previous:
            p_tokens = set(prev.lower().split())
            if q_tokens and p_tokens and len(q_tokens & p_tokens) / len(q_tokens | p_tokens) > 0.7:
                return "<Repeated>"
        return "<Normal>"

    def _extract_token_sequences(self, last_ai_msg, messages, turn_count: int, rollout) -> tuple[list[int], list[int], list[int]]:
        eos = self.tokenizer.eos_token_id or self.tokenizer.pad_token_id or 0
        if last_ai_msg is not None and "prompt_ids" in last_ai_msg.response_metadata:
            prompt_ids = list(last_ai_msg.response_metadata["prompt_ids"])
            response_mask = list(last_ai_msg.response_metadata["response_mask"])
            response_ids = prompt_ids[-len(response_mask):] if response_mask else [eos]
            prompt_ids = prompt_ids[:-len(response_mask)] if response_mask else prompt_ids
        else:
            prompt_ids = self.tokenizer.apply_chat_template(
                _to_openai_messages(messages),
                add_generation_prompt=True,
                tokenize=True,
                enable_thinking=self.doctor_enable_thinking,
            )
            response_ids = [eos]
            response_mask = [1]
        max_resp = rollout.response_length
        max_prompt = rollout.prompt_length
        if len(response_ids) > max_resp:
            response_ids = response_ids[:max_resp]
            response_mask = response_mask[:max_resp]
        if len(prompt_ids) > max_prompt:
            prompt_ids = prompt_ids[-max_prompt:]
        if not response_ids:
            response_ids = [eos]
            response_mask = [1]
        if not prompt_ids:
            prompt_ids = [eos]
        return prompt_ids, response_ids, response_mask
