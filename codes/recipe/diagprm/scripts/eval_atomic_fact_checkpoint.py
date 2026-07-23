#!/usr/bin/env python3
"""Evaluate an AtomicFact/ATPO-style checkpoint on held-out medical QA splits.

This is the "paper-table" style evaluation: run interactive information seeking
on MedQA / MedMCQA / MedicalExam and report final-answer accuracy.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from recipe.diagprm.prompts.atomic_fact import (
    ATOMIC_FACT_DOCTOR_SYSTEM_PROMPT,
    ATOMIC_FACT_PATIENT_SYSTEM_PROMPT,
)

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = lambda x, **kwargs: x


DOCTOR_SYSTEM_PROMPT = """You are a professional medical assistant with strong clinical reasoning and information-seeking skills.

The user provides incomplete patient information, a medical multiple-choice problem, and answer options. Your task is to ask focused questions to collect missing evidence, then choose the correct option.

In each round, decide whether to ask one more question or provide the final answer.

Output format:
- If you need more information, output exactly:
Question: [one specific medical question]
- If you are ready to answer, output exactly:
Final Answer: [one option letter]

Rules:
1. Output either `Question:` or `Final Answer:` and nothing else.
2. Ask exactly one question at a time.
3. Do not repeat questions already asked.
4. Do not reveal hidden reasoning, analysis, chain-of-thought, JSON, markdown, or XML-like tags.
5. Do not output `<think>` or `</think>`.
6. The final answer must be one single option letter from the provided options.
7. Maximum allowed turns: {max_turns}. Current turn: {current_turn} / {max_turns}.
8. On the final turn, you MUST output `Final Answer: [one option letter]`.

/no_think"""


FINAL_SYSTEM_PROMPT = """This is the final allowed turn: {current_turn} / {max_turns}.

You must stop asking questions now and choose the best option from the dialogue history and the original problem.

Output exactly:
Final Answer: [one option letter]

Rules:
1. Do not ask another question.
2. Do not include explanations, uncertainty, JSON, markdown, or hidden reasoning.
3. Do not output `<think>` or `</think>`.
4. The answer must be one single option letter.

/no_think"""


PATIENT_SYSTEM_PROMPT = ATOMIC_FACT_PATIENT_SYSTEM_PROMPT


SPLIT_FILES = {
    "medqa": "atomic_fact_test_medqa.parquet",
    "medmcqa": "atomic_fact_test_medmcqa.parquet",
    "medicalexam": "atomic_fact_test_medicalexam.parquet",
    "all": "atomic_fact_test.parquet",
    "medqa_diag": "atomic_fact_test_medqa_diag.parquet",
    "mediq": "atomic_fact_test_mediq.parquet",
}


SFT_NOTHINK_SYSTEM_PROMPT = ATOMIC_FACT_DOCTOR_SYSTEM_PROMPT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_dir", required=True, help="Merged HF checkpoint directory.")
    parser.add_argument("--base_model", default=None, help="Base model path if model_dir has lora_adapter/.")
    parser.add_argument("--dataset_dir", required=True, help="atomic_fact_rl_v1 directory.")
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["medqa", "medmcqa", "medicalexam"],
        choices=sorted(SPLIT_FILES),
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--max_turns", type=int, default=10)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--doctor_backend",
        choices=["transformers", "vllm"],
        default=os.environ.get("DOCTOR_BACKEND", "transformers"),
    )
    parser.add_argument("--doctor_api_base", default=os.environ.get("DOCTOR_API_BASE", "http://127.0.0.1:8200/v1"))
    parser.add_argument("--doctor_model", default=os.environ.get("DOCTOR_MODEL", "Qwen3-1.7B-RL-step150"))
    parser.add_argument("--doctor_api_key", default=os.environ.get("DOCTOR_API_KEY", ""))
    parser.add_argument("--prompt_style", choices=["ours", "dataset"], default="dataset")
    parser.add_argument(
        "--doctor_enable_thinking",
        default=os.environ.get("ATOMIC_FACT_DOCTOR_THINKING", "1"),
        help="Whether to allow Qwen-style doctor <think> traces. Use 1/0.",
    )
    parser.add_argument("--patient_mode", choices=["rule", "llm"], default="rule")
    parser.add_argument("--patient_api_base", default=os.environ.get("PATIENT_API_BASE", "http://127.0.0.1:8100/v1"))
    parser.add_argument("--patient_model", default=os.environ.get("PATIENT_MODEL", "Qwen3-8B"))
    parser.add_argument("--patient_api_key", default=os.environ.get("PATIENT_API_KEY", ""))
    parser.add_argument("--patient_max_tokens", type=int, default=int(os.environ.get("PATIENT_MAX_TOKENS", "512")))
    parser.add_argument("--save_every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def to_builtin(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): to_builtin(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_builtin(v) for v in value]
    if isinstance(value, np.ndarray):
        return [to_builtin(v) for v in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    return value


def strip_think(text: str) -> str:
    return re.sub(r"<think>[\s\S]*?</think>", "", text or "", flags=re.IGNORECASE).strip()


def normalize_doctor_system_prompt(system: str, doctor_enable_thinking: bool) -> str:
    """Match the no-think SFT system prompt used by atomic_fact_sft_nothink."""
    if doctor_enable_thinking:
        return system
    return SFT_NOTHINK_SYSTEM_PROMPT


CURRENT_TURN_RE = re.compile(
    r"\n+Current turn:\s*\d+\s*/\s*\d+\.\s*$",
    flags=re.IGNORECASE,
)


def set_current_turn_on_latest_user(
    messages: list[dict[str, str]], current_turn: int, max_turns: int
) -> None:
    """Match AtomicFactAgentLoop: update only the latest user message."""
    for message in reversed(messages):
        if message.get("role") != "user":
            continue
        content = CURRENT_TURN_RE.sub("", str(message.get("content", "")).rstrip())
        message["content"] = f"{content}\n\nCurrent turn: {current_turn} / {max_turns}."
        return
    raise ValueError("AtomicFact evaluation requires a user message for the current turn.")


def parse_action(text: str) -> dict[str, str]:
    clean = strip_think(text)
    match = re.search(r"^\s*Final Answer\s*:\s*([A-Z])\b", clean, flags=re.IGNORECASE)
    if match:
        return {"action": "answer", "answer": match.group(1).upper()}
    match = re.search(r"^\s*Question\s*:\s*([\s\S]+?)\s*$", clean, flags=re.IGNORECASE)
    if match:
        return {"action": "ask", "question": match.group(1).strip()}
    return {"action": "invalid", "text": clean}


def load_model(model_dir: Path, base_model: str | None, device: str):
    adapter_dir = model_dir / "lora_adapter"
    if adapter_dir.exists():
        if not base_model:
            raise ValueError("--base_model is required because model_dir contains lora_adapter/")
        from peft import PeftModel

        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
            trust_remote_code=True,
            device_map="auto" if device == "cuda" else None,
        )
        model = PeftModel.from_pretrained(model, str(adapter_dir))
        tokenizer_dir = model_dir
    else:
        model = AutoModelForCausalLM.from_pretrained(
            str(model_dir),
            torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
            trust_remote_code=True,
            device_map="auto" if device == "cuda" else None,
        )
        tokenizer_dir = model_dir
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir), trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    if device != "cuda":
        model.to(device)
    return tokenizer, model


def get_user_content(row: dict[str, Any]) -> str:
    prompt = to_builtin(row.get("prompt", []))
    if isinstance(prompt, list):
        for msg in prompt:
            if isinstance(msg, dict) and msg.get("role") == "user":
                return str(msg.get("content", ""))
    pieces = []
    if row.get("initial_context"):
        pieces.append(str(row["initial_context"]))
    if row.get("question"):
        pieces.append(f"Problem: {row['question']}")
    if row.get("options"):
        pieces.append(f"Options: {row['options']}")
    return "\n".join(pieces)


def get_messages(
    row: dict[str, Any],
    prompt_style: str,
    max_turns: int,
    current_turn: int,
    final: bool = False,
    doctor_enable_thinking: bool = True,
):
    user_content = get_user_content(row)
    if prompt_style == "dataset":
        prompt = to_builtin(row.get("prompt", []))
        if isinstance(prompt, list) and prompt:
            messages = [
                {"role": str(m.get("role", "")), "content": str(m.get("content", ""))}
                for m in prompt
                if isinstance(m, dict) and m.get("role") in {"system", "user", "assistant"}
            ]
            if messages:
                if messages[0].get("role") == "system":
                    messages[0]["content"] = normalize_doctor_system_prompt(
                        messages[0].get("content", ""),
                        doctor_enable_thinking,
                    )
                return messages
    system = FINAL_SYSTEM_PROMPT if final else DOCTOR_SYSTEM_PROMPT
    return [
        {
            "role": "system",
            "content": normalize_doctor_system_prompt(
                system.format(max_turns=max_turns, current_turn=current_turn),
                doctor_enable_thinking,
            ),
        },
        {"role": "user", "content": user_content},
    ]


def runtime_system_prompt(
    base_system: str,
    prompt_style: str,
    max_turns: int,
    current_turn: int,
    final: bool,
    doctor_enable_thinking: bool,
) -> str:
    if prompt_style == "dataset":
        return base_system

    system = FINAL_SYSTEM_PROMPT if final else DOCTOR_SYSTEM_PROMPT
    return system.format(max_turns=max_turns, current_turn=current_turn)


def build_fact_items(row: dict[str, Any]) -> list[dict[str, Any]]:
    extra = to_builtin(row.get("extra_info", {}))
    items = extra.get("atomic_fact_items") if isinstance(extra, dict) else None
    facts = []
    if isinstance(items, list):
        for idx, item in enumerate(items):
            if isinstance(item, dict) and str(item.get("text", "")).strip():
                facts.append({
                    "fact_id": str(item.get("fact_id") or f"F{idx:03d}"),
                    "text": str(item["text"]).strip(),
                    "weight": float(item.get("weight", 1.0)),
                })
    if facts:
        return facts
    raw_facts = to_builtin(row.get("atomic_facts", []))
    if isinstance(raw_facts, list):
        for fact in raw_facts:
            text = str(fact).strip()
            if text:
                facts.append({"fact_id": f"F{len(facts):03d}", "text": text, "weight": 1.0})
    return facts


def rule_patient_answer(facts: list[dict[str, Any]], question: str, used: set[str]) -> dict[str, str]:
    q_words = set(re.findall(r"\b[a-z]{4,}\b", question.lower()))
    q_words -= {
        "patient", "doctor", "symptom", "symptoms", "please", "about", "have", "with",
        "that", "this", "current", "condition", "finding", "relevant", "information",
    }
    best = None
    best_score = 0.0
    for fact in facts:
        words = set(re.findall(r"\b[a-z]{4,}\b", fact["text"].lower()))
        score = len(q_words & words)
        if fact["fact_id"] in used and score > 0:
            score -= 0.25
        if score > best_score:
            best = fact
            best_score = score
    if best and best_score > 0:
        return {"answer": best["text"], "fact_id": best["fact_id"]}
    return {"answer": "The patient cannot answer this question.", "fact_id": "unknown"}


def parse_patient_json(text: str) -> dict[str, str]:
    clean = strip_think(text)
    match = re.search(r"```json\s*([\s\S]*?)```", clean, flags=re.IGNORECASE)
    if match:
        clean = match.group(1)
    else:
        match = re.search(r"(\{[\s\S]*\})", clean)
        clean = match.group(1) if match else clean
    try:
        obj = json.loads(clean)
        if isinstance(obj, dict):
            return {
                "answer": str(obj.get("answer", "")).strip() or "The patient cannot answer this question.",
                "fact_id": str(obj.get("fact_id", "unknown")).strip() or "unknown",
            }
    except Exception:
        pass
    return {"answer": clean or "The patient cannot answer this question.", "fact_id": "unknown"}


def llm_patient_answer(args: argparse.Namespace, facts: list[dict[str, Any]], question: str) -> dict[str, str]:
    facts_text = "\n".join(f"- {f['fact_id']}: {f['text']}" for f in facts) or "(no known facts)"
    payload = {
        "model": args.patient_model,
        "messages": [
            {"role": "system", "content": PATIENT_SYSTEM_PROMPT.format(atomic_facts=facts_text)},
            {"role": "user", "content": question},
        ],
        "max_tokens": args.patient_max_tokens,
        "temperature": 0.0,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    base = args.patient_api_base.rstrip("/")
    url = base + "/chat/completions"
    data = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if args.patient_api_key and not ("127.0.0.1" in base or "localhost" in base):
        headers["Authorization"] = f"Bearer {args.patient_api_key}"
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                obj = json.loads(resp.read().decode("utf-8"))
            content = obj["choices"][0]["message"].get("content", "")
            parsed = parse_patient_json(content)
            valid_ids = {f["fact_id"] for f in facts}
            if parsed["fact_id"] not in valid_ids:
                parsed["fact_id"] = "unknown"
            return parsed
        except (urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError) as exc:
            if attempt == 2:
                return {"answer": f"The patient cannot answer this question.", "fact_id": "unknown"}
            time.sleep(2 ** attempt)
    return {"answer": "The patient cannot answer this question.", "fact_id": "unknown"}


def generate_action(tokenizer, model, messages: list[dict[str, str]], args: argparse.Namespace) -> str:
    doctor_enable_thinking = str(args.doctor_enable_thinking).lower() not in {"0", "false", "no"}
    if args.doctor_backend == "vllm":
        payload = {
            "model": args.doctor_model,
            "messages": messages,
            "max_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "stream": False,
            "chat_template_kwargs": {"enable_thinking": doctor_enable_thinking},
        }
        base = args.doctor_api_base.rstrip("/")
        request = urllib.request.Request(
            base + "/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                **({"Authorization": f"Bearer {args.doctor_api_key}"} if args.doctor_api_key else {}),
            },
            method="POST",
        )
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                with urllib.request.urlopen(request, timeout=180) as response:
                    result = json.loads(response.read().decode("utf-8"))
                return str(result["choices"][0]["message"].get("content", "")).strip()
            except (urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError) as exc:
                last_error = exc
                if attempt < 2:
                    time.sleep(2 ** attempt)
        raise RuntimeError(f"Doctor vLLM request failed after 3 attempts: {last_error}")

    try:
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=doctor_enable_thinking,
        )
    except TypeError:
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    do_sample = args.temperature > 0
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=do_sample,
            temperature=args.temperature if do_sample else None,
            top_p=args.top_p if do_sample else None,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    completion = outputs[0, inputs["input_ids"].shape[1]:]
    return tokenizer.decode(completion, skip_special_tokens=True).strip()


def verify_question(question: str, previous: list[str]) -> str:
    """Match the rule-based verifier used by AtomicFactAgentLoop."""
    if question.count("?") > 1:
        return "<Multiple>"
    question_tokens = set(question.lower().split())
    for previous_question in previous:
        previous_tokens = set(previous_question.lower().split())
        if question_tokens and previous_tokens:
            overlap = len(question_tokens & previous_tokens) / len(question_tokens | previous_tokens)
            if overlap > 0.7:
                return "<Repeated>"
    return "<Normal>"


def run_case(tokenizer, model, row: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    row = to_builtin(row)
    doctor_enable_thinking = str(args.doctor_enable_thinking).lower() not in {"0", "false", "no"}
    answer_idx = str(row.get("answer_idx") or (row.get("reward_model", {}).get("ground_truth", {}) or {}).get("answer", "")).upper()
    facts = build_fact_items(row)
    used_facts: set[str] = set()
    messages = get_messages(
        row,
        args.prompt_style,
        args.max_turns,
        1,
        final=False,
        doctor_enable_thinking=doctor_enable_thinking,
    )
    if not messages or messages[0].get("role") != "system":
        messages = [
            {
                "role": "system",
                "content": normalize_doctor_system_prompt(
                    DOCTOR_SYSTEM_PROMPT.format(max_turns=args.max_turns, current_turn=1),
                    doctor_enable_thinking,
                ),
            },
            *messages,
        ]
    base_system = str(messages[0].get("content", ""))
    try:
        base_system = base_system.format(max_turns=args.max_turns)
    except (KeyError, IndexError):
        pass
    messages[0]["content"] = base_system
    turns = []
    previous_questions: list[str] = []
    final_answer = ""
    invalid = False

    for turn_idx in range(args.max_turns):
        current_turn = turn_idx + 1
        final_turn = current_turn == args.max_turns
        if args.prompt_style != "dataset":
            messages[0] = {
                "role": "system",
                "content": runtime_system_prompt(
                    base_system,
                    args.prompt_style,
                    args.max_turns,
                    current_turn,
                    final_turn,
                    doctor_enable_thinking,
                ),
            }
        else:
            # For dataset prompt_style, still update per-turn placeholders if present
            try:
                messages[0]["content"] = base_system.format(
                    max_turns=args.max_turns, current_turn=current_turn
                )
            except (KeyError, IndexError):
                messages[0]["content"] = base_system
        set_current_turn_on_latest_user(messages, current_turn, args.max_turns)
        raw = generate_action(tokenizer, model, messages, args)
        parsed = parse_action(raw)

        if parsed["action"] == "answer":
            final_answer = parsed["answer"]
            turns.append({"turn_id": turn_idx, "doctor": raw, "action": "answer"})
            messages.append({"role": "assistant", "content": raw})
            break
        if parsed["action"] != "ask":
            invalid = True
            turns.append({"turn_id": turn_idx, "doctor": raw, "action": "invalid"})
            break

        question = parsed["question"]
        verifier = verify_question(question, previous_questions)
        if verifier == "<Repeated>":
            patient = {
                "answer": "You already asked that. Please ask a different question.",
                "fact_id": "unknown",
            }
        elif verifier == "<Multiple>":
            patient = {
                "answer": "Please ask one question at a time.",
                "fact_id": "unknown",
            }
        elif args.patient_mode == "llm":
            patient = llm_patient_answer(args, facts, question)
        else:
            patient = rule_patient_answer(facts, question, used_facts)
        previous_questions.append(question)
        fid = patient.get("fact_id", "unknown")
        if fid != "unknown":
            used_facts.add(fid)
        turns.append({
            "turn_id": turn_idx,
            "doctor": raw,
            "action": "ask",
            "question": question,
            "patient_answer": patient.get("answer", ""),
            "fact_id": fid,
            "verifier": verifier,
        })
        messages.append({"role": "assistant", "content": raw})
        messages.append({"role": "user", "content": patient.get("answer", "The patient cannot answer this question.")})

    correct = bool(final_answer and final_answer.upper() == answer_idx)
    return {
        "uid": row.get("uid", ""),
        "source_name": row.get("source_name", ""),
        "answer_idx": answer_idx,
        "answer_info": row.get("answer_info", ""),
        "final_answer": final_answer,
        "correct": int(correct),
        "has_final": int(bool(final_answer)),
        "invalid": int(invalid),
        "turn_count": len(turns),
        "fact_coverage": len(used_facts) / max(len(facts), 1),
        "n_facts": len(facts),
        "n_used_facts": len(used_facts),
        "turns": turns,
        "doctor_system_prompt": base_system,
        "initial_context": row.get("initial_context", ""),
        "question": row.get("question", ""),
        "options": row.get("options", ""),
    }


def load_split(dataset_dir: Path, split: str, max_samples: int | None, seed: int) -> list[dict[str, Any]]:
    path = dataset_dir / SPLIT_FILES[split]
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_parquet(path)
    if max_samples is not None and max_samples < len(df):
        df = df.sample(n=max_samples, random_state=seed).reset_index(drop=True)
    return [df.iloc[i].to_dict() for i in range(len(df))]


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(results)
    if n == 0:
        return {"n": 0}
    return {
        "n": n,
        "accuracy": sum(r["correct"] for r in results) / n,
        "answer_rate": sum(r["has_final"] for r in results) / n,
        "invalid_rate": sum(r["invalid"] for r in results) / n,
        "avg_turn_count": sum(r["turn_count"] for r in results) / n,
        "avg_fact_coverage": sum(r["fact_coverage"] for r in results) / n,
    }


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.doctor_backend == "vllm":
        tokenizer, model = None, None
    else:
        tokenizer, model = load_model(Path(args.model_dir), args.base_model, args.device)

    all_summary = {}
    for split in args.splits:
        rows = load_split(Path(args.dataset_dir), split, args.max_samples, args.seed)
        results = []
        out_jsonl = out_dir / f"{split}_predictions.jsonl"
        with out_jsonl.open("w", encoding="utf-8") as f:
            for idx, row in enumerate(tqdm(rows, desc=f"eval {split}")):
                result = run_case(tokenizer, model, row, args)
                result["split"] = split
                result["index"] = idx
                results.append(result)
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
                if args.save_every > 0 and (idx + 1) % args.save_every == 0:
                    f.flush()
        summary = summarize(results)
        all_summary[split] = summary
        print(f"[RESULT] {split}: " + json.dumps(summary, ensure_ascii=False))

    (out_dir / "summary.json").write_text(json.dumps(all_summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("[SUMMARY] " + json.dumps(all_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
