#!/usr/bin/env python3
"""
Generate DiagPRM multi-turn SFT data from clean_v2 KG cases.

The generator is intentionally KG-controlled:
  - the hidden diagnosis and fact ids come from the clean_v2 case files;
  - the doctor trajectory asks for selected hidden symptoms, then diagnoses;
  - the patient text is natural language, but the semantic fact remains fixed.

Optional LLM rewriting can be enabled with an OpenAI-compatible chat API. The
LLM is only allowed to paraphrase one planned question/answer pair at a time;
it must not choose facts, change diagnoses, or add new symptoms.

This version supports candidate generation and filtering:
  - generate K candidate trajectories per case;
  - score each candidate by evidence quality and efficiency;
  - keep top-k per case for SFT.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from recipe.diagprm.kg_utils import compute_kg_coverage as recipe_compute_kg_coverage


DEFAULT_SYSTEM_PROMPT = """You are an experienced diagnostic physician conducting a symptom-gathering dialogue. Your goal is to identify the disease through targeted questioning and then provide a confirmed diagnosis.

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


BAD_SYMPTOM_PATTERNS = [
    r"\bserum\b",
    r"\bplasma\b",
    r"\burine\b",
    r"\biga\b",
    r"\bigg\b",
    r"\bigm\b",
    r"\blevels?\b",
    r"\bconcentration\b",
    r"\bmutation\b",
    r"\bgene\b",
    r"\bprotein\b",
    r"\bpeptide\b",
    r"\bcell(s)?\b",
    r"\bchromosome\b",
    r"\bbiopsy\b",
    r"\bct\b",
    r"\bmri\b",
    r"\bultrasound\b",
    r"\bradiograph",
    r"\bx-ray\b",
    r"\bimaging\b",
    r"\bpatholog",
    r"\bhistolog",
    r"\bmetabolic-functional\b",
    r"\bmechanism\b",
]

NON_PATIENT_FACING_PATTERNS = [
    r"\basthma\b",
    r"\bsyndrome\b",
    r"\bdisease\b",
    r"\bpalsy\b",
    r"\bneutrophilia\b",
    r"\bhyperresponsiveness\b",
    r"\bhypophosphatemia\b",
    r"\bfgf\d+\b",
    r"\bradiolucent\b",
    r"\blesions?\b",
    r"\binfants?\b",
    r"\bpreterm\b",
    r"\bnewborn\b",
    r"\boccupational\b",
    r"\bwork-aggravated\b",
    r"\brisk of\b",
    r"\bhigh risk\b",
    r"\bgene\b",
    r"\bmutation\b",
    r"\bpolymorphism\b",
    r"\bchromosome\b",
    r"\bprotein\b",
    r"\bpeptide\b",
    r"\bpathway\b",
    r"\bmechanism\b",
    r"\bstandard definition\b",
    r"\bfirst symptom\b",
    r"\bfirst description\b",
    r"\bpoor outcomes?\b",
    r"\bacquired resistance\b",
    r"\bdiagnos",
]

TEST_RESULT_PATTERNS = [
    r"\bserum\b",
    r"\bplasma\b",
    r"\burine\b",
    r"\biga\b",
    r"\bigg\b",
    r"\bigm\b",
    r"\blevels?\b",
    r"\bconcentration\b",
    r"\bexcretion\b",
    r"\bct\b",
    r"\bmri\b",
    r"\bultrasound\b",
    r"\bradiograph",
    r"\bx-ray\b",
    r"\bimaging\b",
    r"\beeg\b",
    r"\bechocardiography\b",
    r"\bscan\b",
    r"\btest\b",
    r"\blab(oratory)?\b",
    r"\bphosphate\b",
    r"\bblood counts?\b",
]

PRIOR_CLINICAL_FINDING_PATTERNS = [
    r"\bbiopsy\b",
    r"\bpatholog",
    r"\bhistolog",
    r"\bvalve\b",
    r"\bmurmur\b",
    r"\bcardiomegaly\b",
    r"\blymphopenia\b",
    r"\bimmunodeficiency\b",
    r"\bdelayed bone age\b",
    r"\bthymic shadow\b",
    r"\bglottic opening\b",
    r"\bfilum terminale\b",
    r"\bmyelin deficiency\b",
    r"\bcranial nerve\b",
    r"\boculomotor\b",
    r"\bnerve palsy\b",
    r"\bbulbar\b",
    r"\bhospital admission\b",
    r"\bagenesis\b",
    r"\bhypoplasia\b",
    r"\baplasia\b",
    r"\bdysplasia\b",
    r"\bmalformation\b",
]

CAREGIVER_OBSERVABLE_PATTERNS = [
    r"\babnormal (head|feet|face|ear|nose|genital|limb|hand|thumb|toe)\b",
    r"\bdeformit",
    r"\bshort stature\b",
    r"\bdevelopmental delay\b",
    r"\bdelayed development\b",
    r"\bmental retardation\b",
    r"\bintellectual disability\b",
    r"\bearly puberty\b",
    r"\bdelayed puberty\b",
]

PATIENT_TEXT_BAD_PATTERNS = [
    r"\bthe patient\b",
    r"\bbronchial hyperresponsiveness\b",
    r"\bsputum neutrophilia\b",
    r"\bbulbar symptoms?\b",
    r"\bconvulsions\b",
    r"\basthma\b",
    r"\bpalsy\b",
    r"\bhypophosphatemia\b",
    r"\bfgf\d+\b",
    r"\bradiolucent\b",
    r"\blesions?\b",
    r"\bpreterm babies\b",
    r"\bfilum\b",
    r"\bterminale\b",
    r"\bhypoplasia\b",
    r"\bagenesis\b",
    r"\baplasia\b",
    r"\bdysplasia\b",
    r"\bmalformation\b",
    r"\bmorphology\b",
    r"\blymphopenia\b",
    r"\bpolymorphism\b",
    r"\bmutation\b",
    r"\bchromosome\b",
    r"\bprotein\b",
    r"\bpeptide\b",
    r"\bstandard definition\b",
    r"\bfirst description\b",
    r"\bfirst symptom\b",
]


QUESTION_TEMPLATES = [
    "Have you noticed {symptom}?",
    "Do you have {symptom}?",
    "Have you experienced {symptom} recently?",
    "Has there been any sign of {symptom}?",
    "Can you tell me whether {symptom} is present?",
]

ANSWER_TEMPLATES = [
    "Yes, I have noticed {symptom}.",
    "Yes, I have been experiencing {symptom}.",
    "Yes, that has been present.",
    "Yes, the patient reports {symptom}.",
]

OPENING_TEMPLATES = [
    "Hello doctor, I've been experiencing {symptoms}, so I wanted to get checked.",
    "Hi doctor, I came in because I've been having {symptoms}.",
    "Hello, I'm here because I've noticed {symptoms} and I'm concerned.",
    "Hi, I haven't been feeling well. I've been dealing with {symptoms}.",
    "Doctor, I've recently been having {symptoms}, and I'd like to understand what might be going on.",
]


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).strip().lower())


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def is_patient_reportable(symptom: str) -> bool:
    s = normalize_text(symptom)
    if len(s) < 3:
        return False
    return not any(re.search(pattern, s) for pattern in BAD_SYMPTOM_PATTERNS)


def classify_fact_surface(symptom: str) -> str:
    """Classify how a KG fact may be safely surfaced in patient-facing text."""
    s = normalize_text(symptom)
    if len(s) < 3:
        return "not_patient_facing"
    if any(re.search(pattern, s) for pattern in NON_PATIENT_FACING_PATTERNS):
        return "not_patient_facing"
    if any(re.search(pattern, s) for pattern in TEST_RESULT_PATTERNS):
        return "test_result"
    if any(re.search(pattern, s) for pattern in PRIOR_CLINICAL_FINDING_PATTERNS):
        return "prior_clinical_finding"
    if any(re.search(pattern, s) for pattern in CAREGIVER_OBSERVABLE_PATTERNS):
        return "caregiver_observable"
    return "patient_observable"


def is_sft_surface_allowed(surface_type: str) -> bool:
    return surface_type in {
        "patient_observable",
        "caregiver_observable",
        "prior_clinical_finding",
        "test_result",
    }


def is_opening_surface_allowed(surface_type: str) -> bool:
    return surface_type in {"patient_observable", "caregiver_observable"}


def filter_opening_symptoms(initial_symptoms: list[str]) -> list[str]:
    filtered = []
    for symptom in initial_symptoms:
        symptom_norm = normalize_text(symptom)
        if not symptom_norm:
            continue
        if is_opening_surface_allowed(classify_fact_surface(symptom_norm)):
            filtered.append(symptom_norm)
    return filtered


def symptom_df(cases: list[dict[str, Any]]) -> dict[str, int]:
    df: dict[str, int] = {}
    for case in cases:
        for symptom in case.get("symptoms_pool", {}):
            symptom = normalize_text(symptom)
            df[symptom] = df.get(symptom, 0) + 1
    return df


def select_hidden_facts(
    case: dict[str, Any],
    df: dict[str, int],
    n_cases: int,
    min_ask: int,
    max_ask: int,
    rng: random.Random,
) -> list[dict[str, Any]]:
    initial = {normalize_text(x) for x in case.get("initial_symptoms", [])}
    facts = []
    for fact in case.get("symptom_facts", []):
        symptom = normalize_text(fact.get("text", ""))
        if symptom in initial:
            continue
        weight = float(fact.get("weight", 0.0))
        surface_type = classify_fact_surface(symptom)
        if not is_sft_surface_allowed(surface_type):
            continue
        # High disease weight and low global prevalence are useful SFT targets:
        # they teach the model to ask discriminative questions.
        idf = math.log((n_cases + 1.0) / (df.get(symptom, 0) + 1.0))
        score = weight * (1.0 + idf)
        facts.append({**fact, "text": symptom, "surface_type": surface_type, "_score": score})

    if len(facts) < min_ask:
        for fact in case.get("symptom_facts", []):
            symptom = normalize_text(fact.get("text", ""))
            if symptom in initial or any(f["fact_id"] == fact.get("fact_id") for f in facts):
                continue
            surface_type = classify_fact_surface(symptom)
            if not is_sft_surface_allowed(surface_type):
                continue
            weight = float(fact.get("weight", 0.0))
            idf = math.log((n_cases + 1.0) / (df.get(symptom, 0) + 1.0))
            facts.append({**fact, "text": symptom, "surface_type": surface_type, "_score": weight * (1.0 + idf)})

    facts.sort(key=lambda x: (x["_score"], x.get("weight", 0.0)), reverse=True)
    if len(facts) < min_ask:
        return []

    target_n = rng.randint(min_ask, max_ask)
    selected = facts[: min(target_n, len(facts))]
    for fact in selected:
        fact.pop("_score", None)
    return selected


def make_question(symptom: str, rng: random.Random) -> str:
    template = rng.choice(QUESTION_TEMPLATES)
    return template.format(symptom=symptom)


def make_answer(symptom: str, rng: random.Random) -> str:
    template = rng.choice(ANSWER_TEMPLATES)
    return template.format(symptom=symptom)


def question_mentions_diagnosis(question: str, disease: str) -> bool:
    q = normalize_text(question)
    d = normalize_text(disease)
    if not q or not d:
        return False
    return d in q


def is_duplicate_question(question: str, previous_questions: list[str], threshold: float = 0.8) -> bool:
    q_tokens = set(normalize_text(question).split())
    if not q_tokens:
        return True
    for prev in previous_questions:
        p_tokens = set(normalize_text(prev).split())
        if not p_tokens:
            continue
        overlap = len(q_tokens & p_tokens) / max(len(q_tokens | p_tokens), 1)
        if overlap >= threshold:
            return True
    return False


def has_bad_patient_text(text: str) -> bool:
    s = normalize_text(text)
    return any(re.search(pattern, s) for pattern in PATIENT_TEXT_BAD_PATTERNS)


def content_words(text: str) -> set[str]:
    stop = {
        "the", "and", "that", "this", "with", "have", "has", "had", "been",
        "being", "from", "into", "about", "your", "you", "patient", "symptom",
        "present", "noticed", "experiencing", "experience", "doctor", "told",
    }
    return {
        tok for tok in re.findall(r"\b[a-z][a-z0-9-]{3,}\b", normalize_text(text))
        if tok not in stop
    }


def is_verbatim_fact_leak(symptom: str, answer: str, surface_type: str) -> bool:
    if surface_type in {"prior_clinical_finding", "test_result"}:
        return False
    sym_words = content_words(symptom)
    if not sym_words:
        return False
    ans_words = content_words(answer)
    overlap = len(sym_words & ans_words) / max(len(sym_words), 1)
    return overlap >= 0.8


def verify_candidate(
    case: dict[str, Any],
    messages: list[dict[str, str]],
    trace: list[dict[str, Any]],
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    disease = normalize_text(case.get("disease", ""))

    for idx, message in enumerate(messages):
        role = message.get("role")
        content = message.get("content", "")
        if idx == 1 and role == "user" and has_bad_patient_text(content):
            reasons.append("bad_opening_text")
        if role == "assistant":
            try:
                obj = json.loads(content)
            except json.JSONDecodeError:
                reasons.append("assistant_json_invalid")
                continue
            if obj.get("action") not in {"ask", "diagnose"}:
                reasons.append("assistant_action_invalid")
            if obj.get("action") == "ask" and not obj.get("question"):
                reasons.append("assistant_ask_missing_question")
            if obj.get("action") == "diagnose" and not obj.get("diagnosis"):
                reasons.append("assistant_diagnose_missing_diagnosis")
        if idx < len(messages) - 1 and role != "system" and disease and disease in normalize_text(content):
            reasons.append("disease_leak_before_final")

    for item in trace:
        if item.get("action") == "opening":
            opening = str(item.get("text", ""))
            if has_bad_patient_text(opening):
                reasons.append("bad_opening_text")
            for symptom in item.get("initial_symptoms", []):
                if is_verbatim_fact_leak(str(symptom), opening, classify_fact_surface(str(symptom))):
                    reasons.append("opening_verbatim_fact_leak")
            continue
        if item.get("action") != "ask" or item.get("is_noise"):
            continue
        symptom = normalize_text(item.get("symptom", ""))
        answer = str(item.get("answer", ""))
        surface_type = str(item.get("surface_type", "patient_observable"))
        if has_bad_patient_text(answer):
            reasons.append("bad_patient_text")
        if is_verbatim_fact_leak(symptom, answer, surface_type):
            reasons.append("verbatim_fact_leak")

    return not reasons, sorted(set(reasons))


def join_symptoms_naturally(symptoms: list[str]) -> str:
    symptoms = [normalize_text(s) for s in symptoms if normalize_text(s)]
    if not symptoms:
        return "some symptoms"
    if len(symptoms) == 1:
        return symptoms[0]
    if len(symptoms) == 2:
        return f"{symptoms[0]} and {symptoms[1]}"
    return ", ".join(symptoms[:-1]) + f", and {symptoms[-1]}"


def make_opening(initial_symptoms: list[str], rng: random.Random) -> str:
    symptom_text = join_symptoms_naturally(initial_symptoms)
    template = rng.choice(OPENING_TEMPLATES)
    return template.format(symptoms=symptom_text)


def llm_rewrite_opening(
    api_base: str,
    api_key: str,
    model: str,
    initial_symptoms: list[str],
    opening: str,
    timeout: int = 60,
    retries: int = 3,
) -> str:
    url = api_base.rstrip("/") + "/chat/completions"
    symptoms = [normalize_text(s) for s in initial_symptoms if normalize_text(s)]
    prompt = {
        "initial_symptoms": symptoms,
        "opening_draft": opening,
        "constraints": [
            "Return JSON only with key opening.",
            "The opening must be spoken by a patient in first person.",
            "Mention only the provided initial_symptoms.",
            "Do not add any symptom, test result, treatment, or diagnosis.",
            "Do not mention a disease name.",
            "Keep it brief, natural, and conversational.",
            "CRITICAL: Express symptoms in everyday patient language only. Do NOT use medical jargon, Latin terms, or clinical terminology. Describe how the patient feels, not the clinical label (e.g. say 'I keep shaking' not 'convulsions', say 'my head shape looks odd' not 'abnormal head morphology').",
        ],
    }
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "You rewrite a patient's first utterance in a controlled medical dialogue. Preserve the exact symptom facts. /no_think",
            },
            {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
        ],
        "temperature": 0.2,
        "max_tokens": 512,
    }
    body = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    for attempt in range(retries):
        req = urllib.request.Request(url, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            msg = data["choices"][0]["message"]
            content = (msg.get("content") or msg.get("reasoning_content") or "").strip()
            parsed = json.loads(extract_json_object(content))
            rewritten = str(parsed.get("opening", "") if isinstance(parsed, dict) else parsed).strip()
            if rewritten:
                print(f"    [opening] {rewritten[:120]}", flush=True)
                return rewritten
        except urllib.error.HTTPError as e:
            wait = 10.0 * (attempt + 1) if e.code == 429 else 1.5 * (attempt + 1)
            print(f"    [opening] attempt {attempt+1} failed: HTTP {e.code} (wait {wait:.0f}s)", flush=True)
            if attempt == retries - 1:
                break
            time.sleep(wait)
        except (urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError) as e:
            print(f"    [opening] attempt {attempt+1} failed: {e}", flush=True)
            if attempt == retries - 1:
                break
            time.sleep(1.5 * (attempt + 1))
    print(f"    [opening] fallback to template", flush=True)
    return opening


def llm_rewrite_pair(
    api_base: str,
    api_key: str,
    model: str,
    disease: str,
    symptom: str,
    surface_type: str,
    question: str,
    answer: str,
    timeout: int = 60,
    retries: int = 3,
) -> tuple[str, str]:
    url = api_base.rstrip("/") + "/chat/completions"
    prompt = {
        "disease": disease,
        "symptom_fact": symptom,
        "surface_type": surface_type,
        "doctor_question_draft": question,
        "patient_answer_draft": answer,
        "constraints": [
            "Return JSON only with keys question and answer.",
            "The doctor question must ask about exactly the provided symptom_fact.",
            "The patient answer must affirm the provided symptom_fact.",
            "Do not mention the disease name.",
            "Do not add any symptom, test result, treatment, or diagnosis.",
            "Keep both fields concise and natural.",
            "DOCTOR question: ask in plain clinical language a real doctor uses with patients (e.g. 'Have you had any episodes where your hands or body shake uncontrollably?' not 'Have you noticed convulsions?').",
            "PATIENT answer: reply in plain everyday language a non-medical person would use. Do NOT repeat the symptom_fact word-for-word or use medical/Latin terminology. Describe the experience naturally (e.g. 'Yes, I get these shaking fits sometimes' not 'Yes, I have been experiencing convulsions.').",
            "Use surface_type carefully: patient_observable means the patient can feel it; caregiver_observable means the patient/family can notice it; prior_clinical_finding means a doctor previously told the patient; test_result means a test/scan showed it. Do not say the patient 'feels' a test result or anatomical finding.",
        ],
    }
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "You rewrite controlled medical dialogue turns. Preserve the exact medical fact. /no_think",
            },
            {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
        ],
        "temperature": 0.2,
        "max_tokens": 512,
    }
    body = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    for attempt in range(retries):
        req = urllib.request.Request(url, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            msg = data["choices"][0]["message"]
            content = (msg.get("content") or msg.get("reasoning_content") or "").strip()
            parsed = json.loads(extract_json_object(content))
            if isinstance(parsed, dict):
                q = str(parsed.get("question", "")).strip()
                a = str(parsed.get("answer", "")).strip()
            else:
                q, a = "", ""
            if q and a and disease.lower() not in (q + " " + a).lower():
                print(f"    [pair/{symptom[:30]}] Q: {q[:80]}", flush=True)
                print(f"    [pair/{symptom[:30]}] A: {a[:80]}", flush=True)
                return q, a
        except urllib.error.HTTPError as e:
            wait = 10.0 * (attempt + 1) if e.code == 429 else 1.5 * (attempt + 1)
            print(f"    [pair/{symptom[:30]}] attempt {attempt+1} failed: HTTP {e.code} (wait {wait:.0f}s)", flush=True)
            if attempt == retries - 1:
                break
            time.sleep(wait)
        except (urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError) as e:
            print(f"    [pair/{symptom[:30]}] attempt {attempt+1} failed: {e}", flush=True)
            if attempt == retries - 1:
                break
            time.sleep(1.5 * (attempt + 1))
    print(f"    [pair/{symptom[:30]}] fallback to template", flush=True)
    return question, answer


def llm_generate_question(
    api_base: str,
    api_key: str,
    model: str,
    disease: str,
    chief_complaint: str,
    initial_symptoms: list[str],
    collected_symptoms: list[str],
    target_symptom: str,
    previous_questions: list[str],
    timeout: int = 60,
    retries: int = 3,
) -> str:
    url = api_base.rstrip("/") + "/chat/completions"
    prompt = {
        "chief_complaint": chief_complaint,
        "initial_symptoms": initial_symptoms,
        "collected_symptoms": collected_symptoms,
        "target_symptom": target_symptom,
        "previous_questions": previous_questions,
        "constraints": [
            "Return JSON only with key question.",
            "Ask exactly one focused medical question.",
            "The question must target the provided target_symptom.",
            "Do not mention the disease name.",
            "Do not repeat or paraphrase previous_questions.",
            "Do not ask about multiple symptoms in one question.",
            "Keep it natural and concise.",
            "Frame the question in plain language a patient can understand, without using technical jargon or Latin terms. Ask about the patient's experience, not the clinical label (e.g. 'Have you had any episodes where your body shakes or twitches?' not 'Have you experienced convulsions?').",
        ],
    }
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "You are generating one doctor's follow-up question for a controlled diagnostic dialogue. Preserve the target medical fact. /no_think",
            },
            {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
        ],
        "temperature": 0.7,
        "max_tokens": 512,
        "stream": False,
    }
    body = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    for attempt in range(retries):
        req = urllib.request.Request(url, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            msg = data["choices"][0]["message"]
            content = (msg.get("content") or msg.get("reasoning_content") or "").strip()
            parsed = json.loads(extract_json_object(content))
            if isinstance(parsed, dict):
                question = str(parsed.get("question", "")).strip()
            else:
                # LLM returned a plain string instead of JSON object
                question = str(parsed).strip()
            if (
                question
                and not question_mentions_diagnosis(question, disease)
                and not is_duplicate_question(question, previous_questions)
            ):
                print(f"    [genQ/{target_symptom[:30]}] {question[:100]}", flush=True)
                return question
        except urllib.error.HTTPError as e:
            wait = 10.0 * (attempt + 1) if e.code == 429 else 1.5 * (attempt + 1)
            print(f"    [genQ/{target_symptom[:30]}] attempt {attempt+1} failed: HTTP {e.code} (wait {wait:.0f}s)", flush=True)
            if attempt == retries - 1:
                break
            time.sleep(wait)
        except (urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError) as e:
            print(f"    [genQ/{target_symptom[:30]}] attempt {attempt+1} failed: {e}", flush=True)
            if attempt == retries - 1:
                break
            time.sleep(1.5 * (attempt + 1))
    print(f"    [genQ/{target_symptom[:30]}] fallback (empty)", flush=True)
    return ""


def strip_thinking(text: str) -> str:
    """Remove <think>...</think> blocks emitted by Qwen3 thinking mode."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return text.strip()


def extract_json_object(text: str) -> str:
    text = strip_thinking(text)
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text.strip(), flags=re.I).strip()
        text = re.sub(r"```$", "", text).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end >= start:
        return text[start : end + 1]
    return text


def compute_case_coverage(case: dict[str, Any], fact_ids: list[str], weighted: bool = True) -> float:
    symptom_facts = case.get("symptom_facts", [])
    fact_id_to_text = {
        str(f.get("fact_id", "")): normalize_text(f.get("text", ""))
        for f in symptom_facts
        if f.get("fact_id")
    }
    collected = {
        fact_id_to_text[fid]
        for fid in fact_ids
        if fid in fact_id_to_text and fact_id_to_text[fid]
    }
    initial = {normalize_text(x) for x in case.get("initial_symptoms", [])}
    all_collected = collected | initial
    symptoms_pool = case.get("symptoms_pool", {})
    disease = normalize_text(case.get("disease", ""))
    kg = {disease: {normalize_text(k): float(v) for k, v in symptoms_pool.items()}}
    return recipe_compute_kg_coverage(all_collected, disease, kg, weighted=weighted)


def score_candidate(
    case: dict[str, Any],
    trace: list[dict[str, Any]],
    repeated_questions: int,
    invalid_questions: int,
    weighted: bool = True,
) -> dict[str, Any]:
    ask_turns = [t for t in trace if t.get("action") == "ask"]
    diagnose_turns = [t for t in trace if t.get("action") == "diagnose"]
    asked_fact_ids = [str(t.get("fact_id", "")) for t in ask_turns if t.get("fact_id")]
    unique_fact_ids = []
    seen = set()
    for fid in asked_fact_ids:
        if fid and fid not in seen:
            seen.add(fid)
            unique_fact_ids.append(fid)

    final_diag = diagnose_turns[-1].get("diagnosis", "") if diagnose_turns else ""
    gt = normalize_text(case.get("disease", ""))
    final_correct = normalize_text(final_diag) == gt
    coverage = compute_case_coverage(case, unique_fact_ids, weighted=weighted)
    num_new_facts = len(unique_fact_ids)
    num_turns = len(ask_turns) + len(diagnose_turns)
    sum_delta = coverage

    score = (
        3.0 * float(final_correct)
        + 2.0 * coverage
        + 1.0 * num_new_facts
        + 0.5 * sum_delta
        - 0.2 * num_turns
        - 1.0 * repeated_questions
        - 1.0 * invalid_questions
    )

    return {
        "score": score,
        "final_correct": final_correct,
        "kg_coverage": coverage,
        "num_new_facts": num_new_facts,
        "num_turns": num_turns,
        "repeated_questions": repeated_questions,
        "invalid_questions": invalid_questions,
        "asked_fact_ids": unique_fact_ids,
    }


def make_messages(
    case: dict[str, Any],
    facts: list[dict[str, Any]],
    max_turns: int,
    rng: random.Random,
    use_llm: bool,
    api_base: str,
    api_key: str,
    model: str,
    teacher_generate_questions: bool,
    teacher_temperature_tag: int,
) -> tuple[list[dict[str, str]], list[dict[str, Any]], dict[str, Any]]:
    messages = [{"role": "system", "content": DEFAULT_SYSTEM_PROMPT}]
    opening_symptoms = filter_opening_symptoms(case.get("initial_symptoms", []))
    if not opening_symptoms:
        raise ValueError("no patient-facing opening symptoms")
    opening = make_opening(opening_symptoms, rng)
    if use_llm:
        opening = llm_rewrite_opening(
            api_base=api_base,
            api_key=api_key,
            model=model,
            initial_symptoms=opening_symptoms,
            opening=opening,
        )
    messages.append({"role": "user", "content": opening})

    trace = [{"turn_id": 0, "action": "opening", "initial_symptoms": opening_symptoms, "text": opening}]
    previous_questions: list[str] = []
    collected_symptoms = [normalize_text(x) for x in case.get("initial_symptoms", [])]
    repeated_questions = 0
    invalid_questions = 0

    # Build the list of turns: real facts + optionally 1 random-noise turn (20% prob)
    # The noise turn asks an unrelated question the patient cannot confirm — teaching the
    # doctor to handle negative answers and not over-rely on confirmations.
    shuffled_facts = list(facts)
    rng.shuffle(shuffled_facts)

    # Possibly inject one random-noise turn at a random position
    noise_turn_inserted = False
    if rng.random() < 0.20:
        # Pick an off-target symptom from the case's full symptoms_pool that is NOT
        # in the selected hidden facts and is patient-reportable
        selected_ids = {f["fact_id"] for f in shuffled_facts}
        initial_set = {normalize_text(x) for x in case.get("initial_symptoms", [])}
        all_pool = [
            normalize_text(s)
            for s in case.get("symptoms_pool", {})
            if normalize_text(s) not in initial_set
            and classify_fact_surface(normalize_text(s)) == "patient_observable"
        ]
        # Remove symptoms that overlap with selected facts
        selected_texts = {normalize_text(f["text"]) for f in shuffled_facts}
        noise_candidates = [s for s in all_pool if s not in selected_texts]
        if noise_candidates:
            noise_symptom = rng.choice(noise_candidates)
            noise_fact = {"fact_id": "NOISE", "text": noise_symptom, "weight": 0.0, "surface_type": "patient_observable"}
            insert_pos = rng.randint(0, len(shuffled_facts))
            shuffled_facts.insert(insert_pos, noise_fact)
            noise_turn_inserted = True

    for turn_id, fact in enumerate(shuffled_facts, start=1):
        symptom = normalize_text(fact["text"])
        surface_type = str(fact.get("surface_type") or classify_fact_surface(symptom))
        is_noise = fact.get("fact_id") == "NOISE"

        question = ""
        if teacher_generate_questions and use_llm:
            question = llm_generate_question(
                api_base=api_base,
                api_key=api_key,
                model=model,
                disease=str(case["disease"]),
                chief_complaint=opening,
                initial_symptoms=case.get("initial_symptoms", []),
                collected_symptoms=collected_symptoms,
                target_symptom=symptom,
                previous_questions=previous_questions,
            )
        if not question:
            question = make_question(symptom, rng)

        if is_noise:
            # Noise turn: patient answers negatively (they don't have this symptom)
            NOISE_NEGATIVE_ANSWERS = [
                "No, I haven't noticed that.",
                "No, that doesn't seem to be an issue for me.",
                "No, I don't think I have that.",
                "Not that I'm aware of.",
                "No, nothing like that.",
            ]
            answer = rng.choice(NOISE_NEGATIVE_ANSWERS)
        else:
            answer = make_answer(symptom, rng)

        if use_llm and not is_noise:
            question, answer = llm_rewrite_pair(
                api_base=api_base,
                api_key=api_key,
                model=model,
                disease=str(case["disease"]),
                symptom=symptom,
                surface_type=surface_type,
                question=question,
                answer=answer,
            )
        elif use_llm and is_noise:
            # For noise turns rewrite only the question (keep negative answer template)
            question_only, _ = llm_rewrite_pair(
                api_base=api_base,
                api_key=api_key,
                model=model,
                disease=str(case["disease"]),
                symptom=symptom,
                surface_type=surface_type,
                question=question,
                answer=answer,
            )
            question = question_only

        if question_mentions_diagnosis(question, str(case["disease"])):
            invalid_questions += 1
            continue
        if is_duplicate_question(question, previous_questions):
            repeated_questions += 1
            continue

        doctor = {"action": "ask", "question": question}
        messages.append({"role": "assistant", "content": json.dumps(doctor, ensure_ascii=False)})
        messages.append({"role": "user", "content": answer})
        trace.append(
            {
                "turn_id": turn_id,
                "action": "ask",
                "fact_id": fact["fact_id"],
                "symptom": symptom,
                "surface_type": surface_type,
                "question": question,
                "answer": answer,
                "is_noise": is_noise,
            }
        )
        previous_questions.append(question)
        if not is_noise:
            collected_symptoms.append(symptom)

    final = {"action": "diagnose", "diagnosis": normalize_text(case["disease"])}
    messages.append({"role": "assistant", "content": json.dumps(final, ensure_ascii=False)})
    trace.append(
        {
            "turn_id": len(facts) + 1,
            "action": "diagnose",
            "diagnosis": normalize_text(case["disease"]),
        }
    )

    if len([m for m in messages if m["role"] == "assistant"]) > max_turns:
        raise ValueError("assistant turns exceed max_turns")
    metrics = score_candidate(
        case=case,
        trace=trace,
        repeated_questions=repeated_questions,
        invalid_questions=invalid_questions,
    )
    metrics["teacher_temperature_tag"] = teacher_temperature_tag
    return messages, trace, metrics


def dedupe_signature(trace: list[dict[str, Any]]) -> tuple[str, ...]:
    ask_turns = [t for t in trace if t.get("action") == "ask"]
    return tuple(str(t.get("fact_id", "")) for t in ask_turns if t.get("fact_id"))


def generate_case_candidates(
    case: dict[str, Any],
    df: dict[str, int],
    n_cases: int,
    min_ask: int,
    max_ask: int,
    max_turns: int,
    rng: random.Random,
    use_llm: bool,
    api_base: str,
    api_key: str,
    model: str,
    candidates_per_case: int,
    top_k_per_case: int,
    teacher_generate_questions: bool,
    min_kg_coverage: float,
    min_new_facts: int,
) -> list[dict[str, Any]]:
    candidates = []
    for cand_idx in range(candidates_per_case):
        if cand_idx > 0:
            time.sleep(2.0)  # avoid 429 between candidates
        facts = select_hidden_facts(case, df, n_cases, min_ask, max_ask, rng)
        if not facts:
            continue
        try:
            messages, trace, metrics = make_messages(
                case=case,
                facts=facts,
                max_turns=max_turns,
                rng=rng,
                use_llm=use_llm,
                api_base=api_base,
                api_key=api_key,
                model=model,
                teacher_generate_questions=teacher_generate_questions,
                teacher_temperature_tag=cand_idx,
            )
        except ValueError:
            continue

        is_valid, quality_reasons = verify_candidate(case, messages, trace)
        metrics["quality_reasons"] = quality_reasons
        if not is_valid:
            continue

        if not metrics["final_correct"]:
            continue
        if metrics["kg_coverage"] < min_kg_coverage:
            continue
        if metrics["num_new_facts"] < min_new_facts:
            continue

        candidates.append(
            {
                "messages": messages,
                "trace": trace,
                "metrics": metrics,
                "signature": dedupe_signature(trace),
            }
        )

    candidates.sort(
        key=lambda x: (
            x["metrics"]["score"],
            x["metrics"]["kg_coverage"],
            x["metrics"]["num_new_facts"],
            -x["metrics"]["num_turns"],
        ),
        reverse=True,
    )

    selected = []
    seen_signatures = set()
    for cand in candidates:
        if cand["signature"] in seen_signatures:
            continue
        selected.append(cand)
        seen_signatures.add(cand["signature"])
        if len(selected) >= top_k_per_case:
            break
    return selected


def generate_split(
    input_path: Path,
    output_dir: Path,
    split_name: str,
    max_cases: int | None,
    min_ask: int,
    max_ask: int,
    max_turns: int,
    seed: int,
    use_llm: bool,
    api_base: str,
    api_key: str,
    model: str,
    candidates_per_case: int,
    top_k_per_case: int,
    teacher_generate_questions: bool,
    min_kg_coverage: float,
    min_new_facts: int,
    case_stride: int,
    case_offset: int,
) -> dict[str, Any]:
    rng = random.Random(seed)
    cases = load_jsonl(input_path)
    rng.shuffle(cases)
    df = symptom_df(cases)
    n_cases = len(cases)
    if case_stride < 1:
        raise ValueError("case_stride must be >= 1")
    if case_offset < 0 or case_offset >= case_stride:
        raise ValueError("case_offset must be in [0, case_stride)")
    cases = [case for idx, case in enumerate(cases) if idx % case_stride == case_offset]
    if max_cases is not None:
        cases = cases[:max_cases]

    rows = []
    skipped = 0
    selected_cases = 0
    n_total = len(cases)
    for case_idx, case in enumerate(cases, start=1):
        disease_name = case.get("disease", "?")[:40]
        print(f"[{split_name}] {case_idx}/{n_total}  {disease_name}", flush=True)
        selected_candidates = generate_case_candidates(
            case=case,
            df=df,
            n_cases=n_cases,
            min_ask=min_ask,
            max_ask=max_ask,
            max_turns=max_turns,
            rng=rng,
            use_llm=use_llm,
            api_base=api_base,
            api_key=api_key,
            model=model,
            candidates_per_case=candidates_per_case,
            top_k_per_case=top_k_per_case,
            teacher_generate_questions=teacher_generate_questions,
            min_kg_coverage=min_kg_coverage,
            min_new_facts=min_new_facts,
        )
        if not selected_candidates:
            skipped += 1
            print(f"  → skipped (no valid candidates)", flush=True)
            continue
        selected_cases += 1
        best = selected_candidates[0]["metrics"]
        print(
            f"  → ok  score={best['score']:.1f}  cov={best['kg_coverage']:.2f}"
            f"  facts={best['num_new_facts']}  turns={best['num_turns']}"
            f"  correct={best['final_correct']}",
            flush=True,
        )
        for rank, candidate in enumerate(selected_candidates, start=1):
            rows.append(
                {
                    "messages": candidate["messages"],
                    "enable_thinking": False,
                    "data_source": "diagprm_clean_v2_sft",
                    "split": split_name,
                    "disease": normalize_text(case["disease"]),
                    "disease_id": case.get("disease_id", ""),
                    "disease_group": case.get("disease_group", ""),
                    "initial_symptoms": case.get("initial_symptoms", []),
                    "sft_trace": candidate["trace"],
                    "candidate_rank": rank,
                    "candidate_signature": list(candidate["signature"]),
                    "candidate_metrics": candidate["metrics"],
                }
            )

    print(
        f"\n[{split_name}] done: {selected_cases} ok / {skipped} skipped / {len(rows)} rows → saving...",
        flush=True,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / f"diagprm_sft_{split_name}.jsonl"
    parquet_path = output_dir / f"diagprm_sft_{split_name}.parquet"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    pd.DataFrame(rows).to_parquet(parquet_path, index=False)
    quality = compute_quality_report(rows)
    quality_path = output_dir / f"diagprm_sft_{split_name}_quality.json"
    with quality_path.open("w", encoding="utf-8") as f:
        json.dump(quality, f, ensure_ascii=False, indent=2)

    return {
        "split": split_name,
        "input": str(input_path),
        "jsonl": str(jsonl_path),
        "parquet": str(parquet_path),
        "quality": str(quality_path),
        "written": len(rows),
        "skipped": skipped,
        "selected_cases": selected_cases,
        "use_llm": use_llm,
        "candidates_per_case": candidates_per_case,
        "top_k_per_case": top_k_per_case,
        "case_stride": case_stride,
        "case_offset": case_offset,
    }


def compute_quality_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ask_turns = 0
    disease_leaks = 0
    bad_opening_text = 0
    opening_verbatim_fact_leaks = 0
    bad_patient_text = 0
    verbatim_fact_leaks = 0
    third_person_patient = 0
    surface_counts: dict[str, int] = {}

    for row in rows:
        disease = normalize_text(row.get("disease", ""))
        messages = row.get("messages", [])
        for idx, message in enumerate(messages):
            if (
                idx < len(messages) - 1
                and message.get("role") != "system"
                and disease
                and disease in normalize_text(message.get("content", ""))
            ):
                disease_leaks += 1
            if idx == 1 and message.get("role") == "user" and has_bad_patient_text(message.get("content", "")):
                bad_opening_text += 1

        for item in row.get("sft_trace", []):
            if item.get("action") == "opening":
                opening = str(item.get("text", ""))
                if has_bad_patient_text(opening):
                    bad_opening_text += 1
                for symptom in item.get("initial_symptoms", []):
                    if is_verbatim_fact_leak(str(symptom), opening, classify_fact_surface(str(symptom))):
                        opening_verbatim_fact_leaks += 1
                continue
            if item.get("action") != "ask" or item.get("is_noise"):
                continue
            ask_turns += 1
            surface_type = str(item.get("surface_type", "patient_observable"))
            surface_counts[surface_type] = surface_counts.get(surface_type, 0) + 1
            answer = str(item.get("answer", ""))
            symptom = normalize_text(item.get("symptom", ""))
            if re.search(r"\bthe patient\b", answer, flags=re.I):
                third_person_patient += 1
            if has_bad_patient_text(answer):
                bad_patient_text += 1
            if is_verbatim_fact_leak(symptom, answer, surface_type):
                verbatim_fact_leaks += 1

    denom = max(ask_turns, 1)
    return {
        "num_rows": len(rows),
        "ask_turns": ask_turns,
        "surface_type_counts": surface_counts,
        "disease_leaks": disease_leaks,
        "bad_opening_text": bad_opening_text,
        "opening_verbatim_fact_leaks": opening_verbatim_fact_leaks,
        "bad_patient_text": bad_patient_text,
        "third_person_patient": third_person_patient,
        "verbatim_fact_leaks": verbatim_fact_leaks,
        "bad_patient_text_rate": bad_patient_text / denom,
        "third_person_patient_rate": third_person_patient / denom,
        "verbatim_fact_leak_rate": verbatim_fact_leaks / denom,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--timestamp_output", action="store_true")
    parser.add_argument("--max_train_cases", type=int, default=None)
    parser.add_argument("--max_val_cases", type=int, default=500)
    parser.add_argument("--min_ask", type=int, default=2)
    parser.add_argument("--max_ask", type=int, default=4)
    parser.add_argument("--max_turns", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_llm", action="store_true")
    parser.add_argument("--llm_api_base", default=os.environ.get("SFT_LLM_API_BASE", ""))
    parser.add_argument("--llm_api_key", default=os.environ.get("SFT_LLM_API_KEY", ""))
    parser.add_argument("--llm_model", default=os.environ.get("SFT_LLM_MODEL", ""))
    parser.add_argument("--candidates_per_case", type=int, default=4)
    parser.add_argument("--top_k_per_case", type=int, default=1)
    parser.add_argument("--teacher_generate_questions", action="store_true")
    parser.add_argument("--min_kg_coverage", type=float, default=0.5)
    parser.add_argument("--min_new_facts", type=int, default=2)
    parser.add_argument("--case_stride", type=int, default=1)
    parser.add_argument("--case_offset", type=int, default=0)
    args = parser.parse_args()

    if args.use_llm and (not args.llm_api_base or not args.llm_model):
        raise SystemExit("--use_llm requires --llm_api_base and --llm_model, or SFT_LLM_API_BASE/SFT_LLM_MODEL")
    if args.max_ask + 1 > args.max_turns:
        raise SystemExit("--max_turns must be at least --max_ask + 1 for the final diagnosis turn")
    if args.candidates_per_case < 1:
        raise SystemExit("--candidates_per_case must be >= 1")
    if args.top_k_per_case < 1 or args.top_k_per_case > args.candidates_per_case:
        raise SystemExit("--top_k_per_case must be in [1, candidates_per_case]")
    if args.teacher_generate_questions and not args.use_llm:
        raise SystemExit("--teacher_generate_questions requires --use_llm")

    output_dir = args.output_dir
    if args.timestamp_output:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = output_dir / ts

    summaries = []
    summaries.append(
        generate_split(
            input_path=args.dataset_dir / "kg_train_dataset.jsonl",
            output_dir=output_dir,
            split_name="train",
            max_cases=args.max_train_cases,
            min_ask=args.min_ask,
            max_ask=args.max_ask,
            max_turns=args.max_turns,
            seed=args.seed,
            use_llm=args.use_llm,
            api_base=args.llm_api_base,
            api_key=args.llm_api_key,
            model=args.llm_model,
            candidates_per_case=args.candidates_per_case,
            top_k_per_case=args.top_k_per_case,
            teacher_generate_questions=args.teacher_generate_questions,
            min_kg_coverage=args.min_kg_coverage,
            min_new_facts=args.min_new_facts,
            case_stride=args.case_stride,
            case_offset=args.case_offset,
        )
    )
    summaries.append(
        generate_split(
            input_path=args.dataset_dir / "kg_val_dataset.jsonl",
            output_dir=output_dir,
            split_name="val",
            max_cases=args.max_val_cases,
            min_ask=args.min_ask,
            max_ask=args.max_ask,
            max_turns=args.max_turns,
            seed=args.seed + 1,
            use_llm=args.use_llm,
            api_base=args.llm_api_base,
            api_key=args.llm_api_key,
            model=args.llm_model,
            candidates_per_case=args.candidates_per_case,
            top_k_per_case=args.top_k_per_case,
            teacher_generate_questions=args.teacher_generate_questions,
            min_kg_coverage=args.min_kg_coverage,
            min_new_facts=args.min_new_facts,
            case_stride=args.case_stride,
            case_offset=args.case_offset,
        )
    )

    manifest = {
        "dataset_dir": str(args.dataset_dir),
        "output_dir": str(output_dir),
        "timestamp_output": args.timestamp_output,
        "min_ask": args.min_ask,
        "max_ask": args.max_ask,
        "max_turns": args.max_turns,
        "seed": args.seed,
        "candidates_per_case": args.candidates_per_case,
        "top_k_per_case": args.top_k_per_case,
        "teacher_generate_questions": args.teacher_generate_questions,
        "min_kg_coverage": args.min_kg_coverage,
        "min_new_facts": args.min_new_facts,
        "case_stride": args.case_stride,
        "case_offset": args.case_offset,
        "splits": summaries,
        "notes": [
            "Doctor sees only system prompt, patient natural-language opening, and patient natural-language answers.",
            "Hidden fact ids are stored only in sft_trace for auditing and are not part of messages.",
            "Use only train split for SFT training; val split is for SFT validation.",
            "Each case may generate multiple candidate trajectories; only top-k filtered candidates are kept.",
        ],
    }
    manifest_path = output_dir / "sft_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
