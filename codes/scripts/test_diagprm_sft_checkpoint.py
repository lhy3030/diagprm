#!/usr/bin/env python3
"""Smoke-test a merged DiagPRM SFT checkpoint with a few diagnostic prompts."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


SYSTEM_PROMPT = """You are an experienced diagnostic physician conducting a symptom-gathering dialogue. Your goal is to identify the disease through targeted questioning and then provide a confirmed diagnosis.

You do not have access to external tools in this setting. Use only the patient's chief complaint and the patient's answers during the dialogue.

## Clinical Strategy

The patient's opening is incomplete. Do not diagnose from the initial complaint alone.

- Ask one focused question at a time.
- Prefer questions that reveal new clinically relevant symptoms, associated symptoms, or red flags.
- Avoid repeating information already stated by the patient.
- Continue gathering evidence until the dialogue contains enough supporting symptoms to make a diagnosis.
- Diagnose only after collecting multiple pieces of supporting evidence beyond the opening complaint, or on the final turn.

## Output Format

Output a single JSON object, with no text outside the JSON.

When continuing the consultation:
{"action": "ask", "question": "[single focused question about one specific symptom]"}

When ready to diagnose:
{"action": "diagnose", "diagnosis": "[final diagnosis]"}

Rules:
1. Always output a single valid JSON object.
2. Use only two actions: "ask" or "diagnose".
3. Ask exactly one focused question per turn.
4. Do not include hidden reasoning, hypotheses, or confirmed symptom lists.
5. On the final turn, output "diagnose" with your best diagnosis.

/no_think"""


CASES = [
    "Hello doctor, I've had blurry vision that keeps getting worse, and lights look distorted at night.",
    "Hi doctor, I have severe chest pain that came on suddenly and feels like it is moving through my chest.",
    "Hello, I have been feeling very tired and my stomach has been uncomfortable for a while.",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_dir", required=True, help="Merged HuggingFace checkpoint directory.")
    parser.add_argument("--base_model", default=None, help="Original base model path. Needed for LoRA adapters.")
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--case", action="append", default=None, help="Additional patient opening to test.")
    return parser.parse_args()


def extract_json(text: str) -> tuple[dict | None, str | None]:
    text = text.strip()
    try:
        return json.loads(text), None
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return None, "no JSON object found"
    try:
        return json.loads(match.group(0)), None
    except json.JSONDecodeError as exc:
        return None, f"invalid JSON: {exc}"


def load_model(model_dir: Path, base_model: str | None, device: str):
    adapter_dir = model_dir / "lora_adapter"
    load_dir = str(model_dir)
    tokenizer_dir = str(model_dir)

    if adapter_dir.exists():
        if not base_model:
            raise ValueError("--base_model is required because merged checkpoint contains lora_adapter/")
        from peft import PeftModel

        load_dir = base_model
        tokenizer_dir = str(model_dir)
        model = AutoModelForCausalLM.from_pretrained(
            load_dir,
            torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
            trust_remote_code=True,
            device_map="auto" if device == "cuda" else None,
        )
        model = PeftModel.from_pretrained(model, str(adapter_dir))
    else:
        model = AutoModelForCausalLM.from_pretrained(
            load_dir,
            torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
            trust_remote_code=True,
            device_map="auto" if device == "cuda" else None,
        )

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    if device != "cuda":
        model.to(device)
    return tokenizer, model


def main() -> None:
    args = parse_args()
    model_dir = Path(args.model_dir)
    tokenizer, model = load_model(model_dir, args.base_model, args.device)

    cases = CASES + (args.case or [])
    valid = 0
    for idx, opening in enumerate(cases, start=1):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": opening},
        ]
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
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
        completion_ids = outputs[0, inputs["input_ids"].shape[1] :]
        text = tokenizer.decode(completion_ids, skip_special_tokens=True).strip()
        obj, error = extract_json(text)
        ok = (
            error is None
            and isinstance(obj, dict)
            and obj.get("action") in {"ask", "diagnose"}
            and (("question" in obj) if obj.get("action") == "ask" else ("diagnosis" in obj))
        )
        valid += int(ok)
        print("=" * 80)
        print(f"CASE {idx}")
        print(f"PATIENT: {opening}")
        print(f"RAW: {text}")
        print(f"PARSED_OK: {ok}")
        if obj is not None:
            print(f"JSON: {json.dumps(obj, ensure_ascii=False)}")
        if error:
            print(f"ERROR: {error}")

    print("=" * 80)
    print(f"VALID_JSON_ACTIONS: {valid}/{len(cases)}")


if __name__ == "__main__":
    main()
