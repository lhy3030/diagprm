import json, re
from collections import Counter

records = []
with open('merged_train_dataset.jsonl') as f:
    for line in f:
        line = line.strip()
        if line:
            records.append(json.loads(line))

def get_question(rec):
    if 'question' in rec:
        return rec['question']
    if 'prompt' in rec and isinstance(rec['prompt'], list):
        for msg in rec['prompt']:
            if msg.get('role') == 'user':
                m = re.search(r'Problem:\s*(.*?)(?:\nOptions:|$)', msg['content'], re.DOTALL)
                if m:
                    return m.group(1).strip()
    return ''

DIAG = ['most likely diagnosis', 'likely diagnosis', 'most likely cause', 'which diagnosis',
        'the diagnosis', 'what is the diagnosis', 'most likely responsible', 'most probable diagnosis',
        'best diagnosis', 'likely cause of', 'most likely finding', 'condition is', 'condition best described',
        'condition described', 'her condition', 'his condition', 'the condition', 'diagnosis is',
        'most likely suffered', 'most likely have', 'suffering from']
TREAT = ['treatment', 'management', 'therapy', 'treat', 'drug of choice', 'next step',
         'best initial', 'first line', 'best course', 'most appropriate management',
         'most appropriate treatment', 'best treatment', 'should be treated', 'should receive',
         'given to', 'administered', 'prophylaxis', 'intervention']
MECH = ['mechanism', 'pathophysiology', 'mode of action', 'mechanism of action',
        'most likely due to', 'most likely responsible for', 'most likely result of',
        'caused by', 'reason for', 'why does', 'explain']
PHARMA = ['which drug', 'which medication', 'which antibiotic', 'pharmacology',
          'blocks', 'inhibit', 'receptor', 'side effect of', 'adverse effect',
          'drug used', 'used in treatment']
ETHICAL = ['ethical', 'consent', 'legal', 'disclose', 'report', 'obligation']
PROGNOSIS = ['prognosis', 'complication', 'sequelae', 'outcome', 'expected finding']
PROCEDURE = ['procedure', 'investigation', 'imaging', 'lab test', 'biopsy', 'next test',
             'investigation of choice', 'test of choice', 'gold standard', 'diagnostic test']

categories = Counter()
other_qs = []
for rec in records:
    q = get_question(rec)
    ql = q.lower()
    if any(k in ql for k in DIAG):
        categories['诊断题'] += 1
    elif any(k in ql for k in TREAT):
        categories['治疗/处置题'] += 1
    elif any(k in ql for k in MECH):
        categories['机制题'] += 1
    elif any(k in ql for k in PHARMA):
        categories['药理题'] += 1
    elif any(k in ql for k in PROCEDURE):
        categories['检查/操作题'] += 1
    elif any(k in ql for k in ETHICAL):
        categories['伦理题'] += 1
    elif any(k in ql for k in PROGNOSIS):
        categories['预后/并发症'] += 1
    else:
        categories['其他'] += 1
        other_qs.append(q)

total = sum(categories.values())
print(f'总计: {total} 条\n')
for cat, cnt in categories.most_common():
    print(f'  {cat}: {cnt} ({cnt/total*100:.1f}%)')

# 抽样 "其他" 里的题目，看看到底是什么类型
print(f'\n--- "其他" 类题目随机抽样 30 条 ---')
import random
random.seed(42)
sample = random.sample(other_qs, min(30, len(other_qs)))
for q in sample:
    print(f'  [{q[:150]}]')
