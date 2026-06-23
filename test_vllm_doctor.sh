#!/usr/bin/env bash
# ============================================================================
# 测试远程 vLLM 上的 Doctor prompt 是否正常响应
#
# 用法（在 paracloud 登录节点执行）：
#   bash /data/home/scwb729/run/diagprm/test_vllm_doctor.sh
#
# 覆盖参数：
#   VLLM_URL=http://10.252.14.38:8000/v1  bash test_vllm_doctor.sh
#   MODEL=patient-model                    bash test_vllm_doctor.sh
#   TURNS=3                                bash test_vllm_doctor.sh  # 多轮模拟
# ============================================================================
set -euo pipefail

# ─── 配置 ────────────────────────────────────────────────────────────────────
# 读取 endpoint 文件（vLLM 就绪后由 job_vllm_server.sh 写入）
_ENDPOINT_FILE="/data/run01/scwb729/patient_vllm.endpoint"
VLLM_URL="${VLLM_URL:-$(cat "${_ENDPOINT_FILE}" 2>/dev/null | tr -d '[:space:]')}"
VLLM_URL="${VLLM_URL:-http://10.252.14.38:8000/v1}"

# 自动探测 model name（优先用 served-model-name，fallback 到 /v1/models 第一个）
_MODELS_JSON=$(curl -sf --max-time 10 "${VLLM_URL}/models" 2>/dev/null || echo '{}')
_AUTO_MODEL=$(echo "${_MODELS_JSON}" | python3 -c "
import json,sys
d=json.load(sys.stdin)
models=d.get('data',[])
if models:
    # 优先 patient-model
    for m in models:
        if m['id'] == 'patient-model':
            print('patient-model'); sys.exit(0)
    print(models[0]['id'])
else:
    print('Qwen3-1.7B')
" 2>/dev/null || echo "Qwen3-1.7B")

MODEL="${MODEL:-${_AUTO_MODEL}}"
MAX_TURNS=10
SIMULATE_TURNS="${TURNS:-2}"  # 模拟几轮对话（1=只看Turn1, 2=Turn1+一轮回答）

echo "============================================================"
echo " Test: Doctor Prompt → Remote vLLM"
echo " Endpoint      : ${VLLM_URL}"
echo " Model         : ${MODEL}"
echo " Simulate turns: ${SIMULATE_TURNS}"
echo "============================================================"
echo ""

# ─── [1] 健康检查 ────────────────────────────────────────────────────────────
echo ">>> [1] Health check"
if curl -sf --max-time 5 "${VLLM_URL%/v1}/health" > /dev/null; then
    echo "  ✅ healthy"
else
    echo "  ❌ unhealthy / unreachable: ${VLLM_URL%/v1}/health"
    exit 1
fi

echo ""
echo ">>> [2] Available models"
echo "${_MODELS_JSON}" | python3 -c "
import json,sys
d=json.load(sys.stdin)
for m in d.get('data',[]):
    print(f'  - {m[\"id\"]}  (max_len={m.get(\"max_model_len\",\"?\")})')
" 2>/dev/null || echo "  (failed to list)"
echo ""

# ─── Doctor System Prompt (DOCTOR_SYSTEM_PROMPT_NO_KG) ───────────────────────
read -r -d '' DOCTOR_SYSTEM_PROMPT << 'PROMPT_EOF' || true
You are an experienced diagnostic physician conducting a symptom-gathering dialogue. Your goal is to identify the disease through targeted questioning and then provide a confirmed diagnosis.

You do not have access to external tools in this setting. Use only the patient's chief complaint and the patient's answers during the dialogue.

## Recommended Reasoning Strategy

**Turn 1** (patient describes chief complaint):
- Extract the key symptom(s) from the patient's opening.
- Form a small set of plausible diagnostic hypotheses.
- Ask about one targeted symptom that best discriminates among those hypotheses.

**Turn 2-N** (patient answers your question):
- If the patient confirms a symptom: add it to "confirmed" and update your hypothesis.
- If the patient denies or does not know a symptom that is important for your hypothesis: consider an alternative hypothesis.
- Continue asking one high-value unconfirmed symptom at a time.
- When evidence is sufficient, diagnose.

## Output Format (every turn)

Output a single JSON object — no text outside the JSON block.

When continuing the consultation:
```json
{
  "thought": "[Review confirmed symptoms, current hypothesis, and the next most useful question.]",
  "hypothesis": "[current disease hypothesis]",
  "confirmed": ["confirmed_sym1", "confirmed_sym2"],
  "action": "continue",
  "question": "[single focused question about one specific symptom]"
}
```

When ready to diagnose:
```json
{
  "thought": "[Final reasoning based only on the dialogue evidence.]",
  "hypothesis": "[disease name]",
  "confirmed": ["sym1", "sym2", "sym3"],
  "action": "diagnose",
  "diagnosis": "[final diagnosis]"
}
```

## Rules
1. Always output a single valid JSON object — no extra text outside the JSON block.
2. Update "hypothesis" and "confirmed" every turn.
3. You may switch hypotheses at any turn. When switching, reset "confirmed" to [].
4. Ask exactly one focused question per turn (one specific symptom at a time).
5. Only use "action": "diagnose" when you have sufficient evidence.
6. Maximum allowed turns: __MAX_TURNS__. Current turn: __CURRENT_TURN__ / __MAX_TURNS__.
7. On the final turn, you MUST use "action": "diagnose" with your best hypothesis.

/no_think
PROMPT_EOF

# ─── 调用 vLLM 的通用函数 ────────────────────────────────────────────────────
call_vllm() {
    local system_prompt="$1"
    local messages_json="$2"   # JSON array string

    local payload
    payload=$(python3 -c "
import json, sys
system  = sys.argv[1]
msgs    = json.loads(sys.argv[2])
model   = sys.argv[3]
full_msgs = [{'role': 'system', 'content': system}] + msgs
payload = {
    'model'      : model,
    'messages'   : full_msgs,
    'max_tokens' : 512,
    'temperature': 1.0,
    'top_p'      : 0.9,
    'stream'     : False,
}
print(json.dumps(payload))
" "${system_prompt}" "${messages_json}" "${MODEL}")

    local resp
    resp=$(curl -sf --max-time 60 \
        -H "Content-Type: application/json" \
        -d "${payload}" \
        "${VLLM_URL}/chat/completions" 2>&1) || { echo "CURL_FAILED:${resp}"; return 1; }

    echo "${resp}"
}

# ─── 解析 doctor 响应内容 ────────────────────────────────────────────────────
parse_doctor_response() {
    local raw="$1"
    python3 -c "
import json, sys, re

resp = json.loads(sys.argv[1])
content = resp['choices'][0]['message']['content']
usage   = resp.get('usage', {})
finish  = resp['choices'][0].get('finish_reason', '?')

# 去掉 <think>...</think>
content_clean = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()

print('--- Raw content ---')
print(content)
print()
print(f'--- finish_reason: {finish}  tokens: {usage.get(\"prompt_tokens\",\"?\")}→{usage.get(\"completion_tokens\",\"?\")} ---')
print()

# 尝试解析 JSON
def try_parse(s):
    s = s.strip()
    try:
        return json.loads(s)
    except:
        pass
    m = re.search(r'\{.*\}', s, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except:
            pass
    return None

obj = try_parse(content_clean)
if obj:
    print('✅ JSON valid')
    print(f'  action     : {obj.get(\"action\", \"?\")}')
    print(f'  hypothesis : {obj.get(\"hypothesis\", \"?\")}')
    if obj.get('action') == 'continue':
        print(f'  question   : {obj.get(\"question\", \"?\")}')
        print(f'  confirmed  : {obj.get(\"confirmed\", [])}')
    else:
        print(f'  diagnosis  : {obj.get(\"diagnosis\", \"?\")}')
    print(f'  thought    : {str(obj.get(\"thought\", \"\"))[:150]}...')
else:
    print('❌ JSON parse failed')

# 返回 content_clean 供后续对话使用
print('__CONTENT_CLEAN__:' + content_clean)
" "${raw}"
}

# ─── 提取 clean content（去掉 think 标签后的纯文本） ────────────────────────
extract_clean_content() {
    local raw="$1"
    python3 -c "
import json, re
resp = json.loads(sys.argv[1])
content = resp['choices'][0]['message']['content']
clean = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
print(clean)
import sys
" "${raw}" 2>/dev/null || python3 -c "
import json,sys,re
resp = json.loads(sys.argv[1])
content = resp['choices'][0]['message']['content']
print(re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip())
" "${raw}"
}

# ─── 模拟患者回答（预设脚本，用于多轮测试）────────────────────────────────
# 测试病例：Parinaud syndrome（松果体区病变）
# 症状：双眼复视 + 垂直凝视障碍 + 瞳孔光反射消失 + 视盘水肿
PATIENT_CHIEF_COMPLAINT="Doctor, I've been experiencing double vision in both eyes and I seem to have difficulty moving my eyes upward. It started about two weeks ago and has been getting worse."

PATIENT_TURN2_REPLY="Yes, I have noticed that. My pupils don't seem to react to light normally, they appear fixed and dilated."

PATIENT_TURN3_REPLY="I do have some headaches, and my vision has been blurry. I haven't noticed any drooping eyelid though."

# ─── Turn 1 ──────────────────────────────────────────────────────────────────
CURRENT_TURN=1
echo ">>> [3] Turn ${CURRENT_TURN} — Patient chief complaint"
echo "  👤 Patient: ${PATIENT_CHIEF_COMPLAINT}"
echo ""

SYSTEM_T1="${DOCTOR_SYSTEM_PROMPT//__MAX_TURNS__/${MAX_TURNS}}"
SYSTEM_T1="${SYSTEM_T1//__CURRENT_TURN__/${CURRENT_TURN}}"

MESSAGES_T1=$(P0="${PATIENT_CHIEF_COMPLAINT}" python3 -c "
import json, os
msgs = [{'role': 'user', 'content': os.environ['P0']}]
print(json.dumps(msgs))
")
RESP_T1=$(call_vllm "${SYSTEM_T1}" "${MESSAGES_T1}")

echo "  🩺 Doctor (Turn 1):"
parse_doctor_response "${RESP_T1}"
echo ""

if [ "${SIMULATE_TURNS}" -lt 2 ]; then
    echo "============================================================"
    echo " Done (single turn). Set TURNS=2 or TURNS=3 for more."
    echo "============================================================"
    exit 0
fi

# ─── Turn 2 ──────────────────────────────────────────────────────────────────
CURRENT_TURN=2
DOCTOR_REPLY_T1=$(extract_clean_content "${RESP_T1}")
echo ">>> [4] Turn ${CURRENT_TURN} — Patient answers doctor's question"
echo "  👤 Patient: ${PATIENT_TURN2_REPLY}"
echo ""

SYSTEM_T2="${DOCTOR_SYSTEM_PROMPT//__MAX_TURNS__/${MAX_TURNS}}"
SYSTEM_T2="${SYSTEM_T2//__CURRENT_TURN__/${CURRENT_TURN}}"

# 用环境变量传递内容，避免 bash heredoc/单引号转义问题
MESSAGES_T2=$(P0="${PATIENT_CHIEF_COMPLAINT}" A0="${DOCTOR_REPLY_T1}" P1="${PATIENT_TURN2_REPLY}" python3 -c "
import json, os
msgs = [
    {'role': 'user',      'content': os.environ['P0']},
    {'role': 'assistant', 'content': os.environ['A0']},
    {'role': 'user',      'content': os.environ['P1']},
]
print(json.dumps(msgs))
")

RESP_T2=$(call_vllm "${SYSTEM_T2}" "${MESSAGES_T2}")

echo "  🩺 Doctor (Turn 2):"
parse_doctor_response "${RESP_T2}"
echo ""

if [ "${SIMULATE_TURNS}" -lt 3 ]; then
    echo "============================================================"
    echo " Done (2 turns). Set TURNS=3 for one more turn."
    echo "============================================================"
    exit 0
fi

# ─── Turn 3 ──────────────────────────────────────────────────────────────────
CURRENT_TURN=3
DOCTOR_REPLY_T2=$(extract_clean_content "${RESP_T2}")
echo ">>> [5] Turn ${CURRENT_TURN} — Patient answers again"
echo "  👤 Patient: ${PATIENT_TURN3_REPLY}"
echo ""

SYSTEM_T3="${DOCTOR_SYSTEM_PROMPT//__MAX_TURNS__/${MAX_TURNS}}"
SYSTEM_T3="${SYSTEM_T3//__CURRENT_TURN__/${CURRENT_TURN}}"

MESSAGES_T3=$(P0="${PATIENT_CHIEF_COMPLAINT}" A0="${DOCTOR_REPLY_T1}" P1="${PATIENT_TURN2_REPLY}" A1="${DOCTOR_REPLY_T2}" P2="${PATIENT_TURN3_REPLY}" python3 -c "
import json, os
msgs = [
    {'role': 'user',      'content': os.environ['P0']},
    {'role': 'assistant', 'content': os.environ['A0']},
    {'role': 'user',      'content': os.environ['P1']},
    {'role': 'assistant', 'content': os.environ['A1']},
    {'role': 'user',      'content': os.environ['P2']},
]
print(json.dumps(msgs))
")

RESP_T3=$(call_vllm "${SYSTEM_T3}" "${MESSAGES_T3}")

echo "  🩺 Doctor (Turn 3):"
parse_doctor_response "${RESP_T3}"
echo ""

echo "============================================================"
echo " Done. Run with TURNS=N (max 3) to extend simulation."
echo "============================================================"
