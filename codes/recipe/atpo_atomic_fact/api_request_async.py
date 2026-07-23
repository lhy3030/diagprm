import aiohttp
import asyncio
import json
import os
import time

async def request_vllm_async(content_str, apikey, model="Qwen3-8B", env='test',
                       session_id=None, temperature=0.8, max_token=1024, do_sample=True,
                       top_p=1.0, top_k=-1, max_try=5, retry_interval=1):

    openai_base = os.environ.get("PATIENT_API_BASE") or os.environ.get("ATPO_PATIENT_API_BASE")
    if openai_base:
        url = openai_base.rstrip("/")
        if url.endswith("/v1"):
            url = f"{url}/completions"
        elif not url.endswith("/completions"):
            url = f"{url}/v1/completions"
        authorization = 'Bearer ' + (os.environ.get("PATIENT_API_KEY") or apikey or "EMPTY")
        headers = {
            'Content-Type': 'application/json',
            'Authorization': authorization,
        }
        body = {
            "model": os.environ.get("PATIENT_MODEL") or os.environ.get("ATPO_PATIENT_MODEL") or model,
            "prompt": content_str,
            "max_tokens": max_token,
            "temperature": temperature if do_sample else 0.0,
            "top_p": top_p,
            "stream": False,
        }
        if top_k is not None and int(top_k) >= 0:
            body["top_k"] = top_k

        async with aiohttp.ClientSession() as session:
            for attempt in range(1, max_try + 1):
                try:
                    async with session.post(
                        url,
                        data=json.dumps(body, ensure_ascii=False).encode('utf-8'),
                        headers=headers,
                    ) as resp:
                        if resp.status != 200:
                            text = await resp.text()
                            print(f"[Try {attempt}/{max_try}] OpenAI-compatible request failed: HTTP {resp.status}: {text[:300]}")
                        else:
                            data = await resp.json(content_type=None)
                            choice = data["choices"][0]
                            res = choice.get("text")
                            if res is None:
                                res = choice.get("message", {}).get("content")
                            if res and res.strip():
                                return res
                            print(f"[Try {attempt}/{max_try}], return content is empty (space or empty string)")
                except Exception as e:
                    print(f"[Try {attempt}/{max_try}] OpenAI-compatible request error: {e}")

                if attempt < max_try:
                    await asyncio.sleep(retry_interval)

        return None

    url = 'http://xxxxxxxxxxxxxxxxxxxxxxxxxxx'

    authorization = 'Bearer ' + apikey
    headers = {
        'Content-Type': 'application/json',
        'Authorization': authorization
    }

    body = {
        "session_id": "hf-xxxxxxxxxxxxxxxxxxxxxxxxx" if session_id is None else session_id,
        "request_id": "hf-xxxxxxxxxxxxxxxxxxxxxxxxx",
        "model": model,
        "prompt": content_str,
        "source": {
            "ori_query": "1234"
        },
        'extra_args': {
            'max_new_tokens': max_token,
            'top_p': top_p,
            'top_k': top_k,
            'logprobs': True,
            'temperature': temperature,
            'top_p_decay': 0.0,
            'top_p_bound': 0.0,
            'add_BOS': False,
            'stop_on_double_eol': False,
            'stop_on_eol': False,
            'prevent_newline_after_colon': False,
            'random_seed': 42,
            'no_log': True,
            'stop_token': 50256,
            'length_penalty': 1.0,
            'do_sample': do_sample,
            'no_repeat_ngram_size': 0,
            'beam_width': 1
        }
    }

    async with aiohttp.ClientSession() as session:
        for attempt in range(1, max_try + 1):
            try:
                async with session.post(
                    url,
                    data=json.dumps(body, ensure_ascii=False).encode('utf-8'),
                    headers=headers
                ) as resp:
                    if resp.status != 200:
                        print(f"[Try {attempt}/{max_try}] Request failed: HTTP {resp.status}")
                    else:
                        data = await resp.json(content_type=None)
                        res = data["choices"][0]["message"]["content"]
                        
                        if res and res.strip():
                            return res
                        else:
                            print(f"[Try {attempt}/{max_try}], return content is empty (space or empty string)")
            except Exception as e:
                print(f"[Try {attempt}/{max_try}] request vllm error: {e}")

            if attempt < max_try:
                await asyncio.sleep(retry_interval)

    return None
