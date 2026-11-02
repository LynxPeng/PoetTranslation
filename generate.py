"""配置驱动翻译；标准库实现，不在代码或输出中保存API密钥。"""
import argparse
from collections import ChainMap
from contextlib import contextmanager
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
import copy
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
import random
import shutil
from pathlib import Path
import re
import sys
import threading
import time
from urllib import error, request


class ResponseError(ValueError):
    """接口返回不符合非思考、完整结果约定。"""


class ThinkingError(ResponseError):
    """思考配置异常不得当作正常方法回退。"""


class UnconfirmedCallError(ValueError):
    """已发送但结果未确认的请求保持待处理，不写成最终失败。"""


TRANSLATION_PREAMBLE = (
    "<要求>将翻译后的完整纯文本放入一对<translation>和</translation>标签中，保留诗行换行。</要求>\n\n"
)

TRANSLATION_SCHEMA = {"type": "object", "properties": {"translation": {"type": "string"}},
                      "required": ["translation"], "additionalProperties": False}
TRANSLATION_JSON_MESSAGE = {"role": "system", "content": 'Return JSON only: one object with exactly one string field "translation" containing the complete English translation. Example: {"translation":"English poem text."}'}
_INPLACE_TARGETS = set()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def append_jsonl(path, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    row = dict(row)
    row.setdefault("timestamp", datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="milliseconds"))
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_jsonl_atomic(path, rows):
    """读者或杀毒程序短暂占用Windows文件时，仅重试落盘，不重新调用API。"""
    temporary = Path(path).with_suffix('.replacement.tmp')
    with temporary.open('w', encoding='utf-8') as file:
        for row in rows:file.write(json.dumps(row, ensure_ascii=False) + '\n')
    def inplace():
        data = temporary.read_bytes()
        with Path(path).open('r+b') as target:
            original = target.read()
            try:
                target.seek(0);target.write(data);target.truncate();target.flush();os.fsync(target.fileno())
            except BaseException:
                target.seek(0);target.write(original);target.truncate();target.flush()
                raise
        temporary.unlink()
    target_key = str(Path(path).resolve())
    if target_key in _INPLACE_TARGETS:
        inplace()
        return
    for attempt in range(8):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == 7:
                # Windows查看器可能允许写入却禁止重命名。原文件/尝试已备份，
                # 同进程锁内原位写入；异常时恢复原字节，不重发任何API请求。
                inplace()
                _INPLACE_TARGETS.add(target_key)
                return
            time.sleep(0.1 * (attempt + 1))


def log_event(event, stream=None):
    timestamp = datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="milliseconds")
    print(json.dumps(dict(event, timestamp=timestamp), ensure_ascii=False), file=stream, flush=True)


def read_jsonl(path):
    with Path(path).open(encoding="utf-8-sig") as file:
        for number, line in enumerate(file, 1):
            if line.strip():
                try:row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"JSONL损坏：{path}:{number}：{exc.msg}；请备份核查，不自动跳过") from exc
                yield row


@contextmanager
def generation_lock(path):
    """OS持有锁，进程退出自动释放；线程并发仍由RealtimeSteps协调。"""
    path = Path(path).with_suffix('.lock')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+b') as file:
        if file.seek(0, 2) == 0:
            file.write(b'0');file.flush()
        file.seek(0)
        try:
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(file.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise ValueError('已有生成进程正在使用本项目，请等待或暂停后再启动另一轨道') from exc
        try:yield
        finally:
            file.seek(0)
            if os.name == 'nt':msvcrt.locking(file.fileno(), msvcrt.LK_UNLCK, 1)
            else:fcntl.flock(file.fileno(), fcntl.LOCK_UN)


def existing_ids(path):
    return {row["id"] for row in read_jsonl(path)} if Path(path).exists() else set()


def load_step_cache(path):
    cache = {row['id']: row for row in read_jsonl(path)} if Path(path).exists() else {}
    return {key: row for key, row in cache.items() if row['status'] != 'retry_authorized'}


def load_config(path, batch=False):
    path = Path(path).resolve()
    config = read_json(path)
    experiments = dict(config.get("experiments", {}))
    for fragment in config.get("experiment_files", []):
        fragment_path = (path.parent / fragment).resolve()
        if batch:
            project = Path(__file__).resolve().parent
            try:archived = project / "backups/batch_2026-11-01" / fragment_path.relative_to(project)
            except ValueError:archived = fragment_path
            if archived.exists():fragment_path = archived
        for key, value in read_json(fragment_path)["experiments"].items():
            if key in experiments and experiments[key] != value:
                raise ValueError("条件定义冲突：" + key)
            experiments[key] = value
    config["experiments"] = experiments
    if config.get("result_json_templates"):
        config["output_templates"] = read_json(path.parent / config["result_json_templates"])
    config.pop("experiment_files", None)
    return config, path.parent


def resolve_recipe(config, name, seen=()):
    if name in seen:
        raise ValueError("实验别名循环：" + name)
    recipe = config["experiments"][name]
    return resolve_recipe(config, recipe["alias_of"], seen + (name,)) if "alias_of" in recipe else recipe


def model_name(config, name, primary):
    if name != "partner":
        return name
    pair = config.get("generation_tracks", ["mimo", "qwen"])
    return next(key for key in pair if key != primary)


def material_values(config, root, recipe, poem):
    values = {"notes_token_limit": str(256 if len(re.findall(r"[\u3400-\u9fff]", poem["text"])) <= 70 else
                                      512 if len(re.findall(r"[\u3400-\u9fff]", poem["text"])) <= 200 else 1024)}
    values["explanation_token_limit"] = values["notes_token_limit"]
    for name, path in config.get("materials", {}).items():
        asset = read_json(root / path)
        if name == "examples":
            binding = recipe.get("bindings", {}).get("examples", {"count": 0, "quality": "high"})
            examples = [{"source": {k: item["source"][k] for k in ("title", "author", "text")},
                         "translation": item[binding["quality"]]["translation"]}
                        for item in asset["examples"][:binding["count"]]]
            values["examples"] = json.dumps(examples, ensure_ascii=False)
        else:
            mapping = asset.get("poems", asset)
            data = mapping.get(poem["id"], [])
            if name == "static_references":
                limit = recipe.get("bindings", {}).get("reference", {}).get("max_chars")
                if limit is not None and len(data) > limit:
                    raise ValueError("静态资料超过配置字符预算：" + poem["id"])
            values[{"glossaries": "glossary", "static_references": "reference", "references": "__reference_entries"}.get(name, name)] = data if name == "static_references" and isinstance(data, str) else json.dumps(data, ensure_ascii=False)
    needed = set(re.findall(r"\{\{(\w+)\}\}", json.dumps(recipe["steps"], ensure_ascii=False)))
    if any(s.get("op") == "lookup" for s in recipe["steps"]):needed.add("__reference_entries")
    return {k: v for k, v in values.items() if k in needed}


def model_config(config, name):
    model = dict(config["models"][name])
    for key in ("model", "base_url"):
        if model.get(key + "_env"):
            model[key] = os.environ.get(model[key + "_env"], model.get(key))
    return model


def request_body(model, messages, limit, temperature=0.7, seed=None):
    extra = model.get("extra_body", {})
    if any(key in extra for key in ("model", "messages", "stream", "reasoning_effort", "reasoning", "thinking", "enable_thinking", "thinking_budget", "max_tokens", "max_completion_tokens")):
        raise ValueError("extra_body不能覆盖模型、消息、输出上限或thinking；关闭方式由provider决定")
    provider = model.get("provider")
    enabled = model.get("thinking_enabled", False)
    if type(enabled) is not bool:raise ValueError("thinking_enabled须为布尔值")
    if provider == "deepseek" and enabled:raise ValueError("当前评测端必须关闭thinking")
    budget = model.get("thinking_budget")
    if budget is not None and (provider != "qwen" or not enabled or type(budget) is not int or budget < 1):
        raise ValueError("thinking_budget仅用于开启思考的Qwen，须为正整数；MiMo不支持独立思考预算")
    body = dict(extra, model=model.get("model"), messages=messages,
                temperature=temperature, top_p=1, stream=False)
    if provider in ("mimo", "deepseek"):
        body["thinking"] = {"type": "enabled" if enabled else "disabled"}
        if provider == "mimo" and enabled:
            body.pop("temperature")
            body.pop("top_p")
    elif provider == "qwen":
        body["enable_thinking"] = enabled
        if budget is not None:body["thinking_budget"] = budget
    elif provider == "local" and model.get("nonthinking_verified") is True:
        # 对应服务须先独立验证；不猜测Ollama/vLLM等服务的关闭字段。
        if not extra or "true" in json.dumps(extra).lower():
            raise ValueError("本地非思考配置须提供已核实的关闭参数")
    else:
        raise ValueError("未知provider，或本地关闭thinking尚未验证")
    body["max_completion_tokens" if provider == "mimo" else "max_tokens"] = limit
    if seed is not None and model.get("supports_seed", False):
        body["seed"] = seed
    return body


_RATE_LOCK = threading.Lock()
_RATE_NEXT = {}


def rate_key(model):
    return model.get('base_url'), model.get('model'), model.get('api_key_env')


def wait_for_rate_slot(model):
    rpm = model.get('requests_per_minute', 0)
    while True:
        with _RATE_LOCK:
            now = time.monotonic()
            delay = _RATE_NEXT.get(rate_key(model), now) - now
            if delay <= 0:
                if rpm:_RATE_NEXT[rate_key(model)] = now + 60 / rpm
                else:_RATE_NEXT.pop(rate_key(model), None)
                return
        time.sleep(min(delay, 1))


def rate_cooldown(model, seconds):
    with _RATE_LOCK:
        key = rate_key(model)
        _RATE_NEXT[key] = max(_RATE_NEXT.get(key, 0), time.monotonic() + seconds)


def chat(model, body):
    if not model.get("model") or not model.get("base_url"):
        raise ValueError("请配置模型名及base_url，或其对应环境变量")
    key = os.environ.get(model.get("api_key_env", ""), "")
    if model.get("provider") != "local" and not key:
        raise ValueError(f"缺少环境变量 {model.get('api_key_env')}")
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = "Bearer " + key
    url = model["base_url"].rstrip("/") + "/chat/completions"
    attempts = []
    max_attempts = 6 if model.get('provider') == 'mimo' else 3
    for attempt in range(max_attempts):
        wait_for_rate_slot(model)
        start = time.monotonic()
        try:
            req = request.Request(url, json.dumps(body, ensure_ascii=False).encode(), headers)
            with request.urlopen(req, timeout=model.get("timeout", 180)) as response:
                raw = json.loads(response.read())
            attempts.append({"status": "ok", "seconds": time.monotonic() - start})
            raw["_attempts"] = attempts
            return raw
        except (error.HTTPError, error.URLError, TimeoutError) as exc:
            code = getattr(exc, "code", None)
            detail = ''
            if isinstance(exc, error.HTTPError):
                try:detail = exc.read().decode('utf-8', errors='replace').replace(key, '[redacted]')[:2000]
                except Exception:pass
            attempts.append({"status": "transport_error", "http_status": code,
                             "seconds": time.monotonic() - start, 'server_error': detail})
            # 只重试运输错误，不因结果差或格式失败重新采样。
            if attempt == max_attempts - 1 or (code is not None and code not in (408, 429) and code < 500):
                failure = RuntimeError(f"API运输失败：HTTP {code}；尝试{attempt + 1}次")
                failure.attempts = attempts
                raise failure from exc
            if code == 429:
                try:retry_after = float(exc.headers.get('Retry-After', 0))
                except (TypeError, ValueError, AttributeError):retry_after = 0
                delay = min(60, max(retry_after, 5 * 2 ** attempt) + random.uniform(0, 2))
                attempts[-1]['retry_after_seconds'] = delay
                rate_cooldown(model, delay)
                if model.get('provider') != 'mimo':time.sleep(delay)
            else:time.sleep(min(60, 2 ** attempt + random.uniform(0, 1)))


def response_text(raw, allow_thinking=False, allow_tag_surroundings=False, natural_intermediate=False):
    if not isinstance(raw, dict) or not isinstance(raw.get("choices"), list) or not raw["choices"]:
        raise ResponseError("接口缺少非空choices数组")
    choice = raw["choices"][0]
    if not isinstance(choice, dict) or not isinstance(choice.get("message"), dict):
        raise ResponseError("接口choice/message结构错误")
    message = choice["message"]
    usage = raw.get("usage") or {}
    if not isinstance(usage, dict) or not isinstance(usage.get("completion_tokens_details") or {}, dict):
        raise ResponseError("接口usage结构错误")
    reasoning = (usage.get("completion_tokens_details") or {}).get("reasoning_tokens", 0)
    if not allow_thinking and (message.get("reasoning_content") or reasoning or usage.get("reasoning_tokens", 0)):
        raise ThinkingError("接口返回了思考内容或推理token，停止运行并检查配置")
    if choice.get("finish_reason") != "stop":
        raise ResponseError("接口未正常完成：" + str(choice.get("finish_reason")))
    text = message.get("content")
    if isinstance(text, str) and re.search(r"</?think\s*>", text, re.I) and not (allow_thinking and allow_tag_surroundings):
        if not natural_intermediate:raise ThinkingError("接口输出可见思考标签，停止检查配置")
        text = re.sub(r'<think\s*>.*?</think\s*>', '', text, flags=re.I | re.S)
        if re.search(r'</think\s*>', text, re.I):text = re.split(r'</think\s*>', text, flags=re.I)[-1]
        if re.search(r'</?think\s*>', text, re.I):raise ThinkingError('中间响应思考标签不完整，无法提取正文')
    if not isinstance(text, str) or not text.strip():
        raise ResponseError("空结果")
    return text.strip()


def render(template, values):
    def replace(match):
        name = match.group(1)
        if name not in values:
            raise ValueError("模板引用未定义：" + name)
        return values[name]
    return re.sub(r"\{\{(\w+)\}\}", replace, template)


def stage_body(step, values, model, seed, history=None):
    slot = step.get("sample_slot", step["id"])
    seed = int(digest([seed, slot])[:8], 16) % (2 ** 31)
    limit = step.get("max_tokens", 4096)
    if step.get("max_tokens_by_hanzi"):
        count = len(re.findall(r"[\u3400-\u9fff]", json.loads(values["source"])["text"]))
        limits = step["max_tokens_by_hanzi"]
        limit = next((limits[str(n)] for n in sorted(int(k) for k in limits if k != "default") if count <= n), limits["default"])
    if model.get("provider") == "mimo" and model.get("thinking_enabled", False):
        allowance = model.get("thinking_completion_allowance", 0)
        if type(allowance) is not int or allowance < 0:raise ValueError("thinking_completion_allowance须为非负整数")
        limit += allowance  # 原生总上限的余量，接口不保证思考与答案各自分配。
    content = step.get("output_preamble", "") + render(step["prompt"], values)
    if step.get('exact_copy_from'):
        content += '\n所需JSON已正确转义，请逐字原样返回，不删除标题或其他正文内容：\n' + json.dumps(
            {'translation': values[step['exact_copy_from']].strip()}, ensure_ascii=False)
    messages = [m for m in (history or []) if not (m.get('role') == 'system' and m.get('content', '').startswith('Return JSON only: '))]
    messages.append({"role": "user", "content": content})
    structured = step.get("translation_format") == "json_only" or bool(step.get("native_json_key"))
    active = dict(model)
    if structured:
        field = step.get("native_json_key")
        examples = {'candidate_id': 1, 'score': 0,
                    'ranking': '按质量从好到坏排列的原始ID，逗号分隔，每个ID一次，无解释',
                    'feedback': '简明反馈正文，保留本轮要求的问题信息，不输出思考过程。最多1200个汉字，不逐句复述全文；无问题写无需修改。',
                    'result': '本轮要求的完整结果正文，不输出思考过程或重复段落'}
        messages.insert(0, {"role": "system", "content": 'Return JSON only: ' + json.dumps({field: examples.get(field, '本轮结果正文')}, ensure_ascii=False)} if field else dict(TRANSLATION_JSON_MESSAGE))
        if field == 'feedback':
            messages[0]['content'] = 'Return JSON only: one object with exactly one string field "feedback" containing the requested review text. Example: {"feedback":"Review text."}'
        if active.get("provider") == "qwen":
            active["thinking_enabled"] = False
            active.pop("thinking_budget", None)
    body = request_body(active, messages, limit, temperature=step.get("temperature", 0.7), seed=seed)
    if structured:body["response_format"] = {"type": "json_object"}
    return body


def generation_json(text):
    """接受完整result_json标签、裸JSON或单个代码块，拒绝额外内容。"""
    tagged = re.fullmatch(r"<result_json>\s*(.*?)\s*</result_json>", text.strip(), re.DOTALL)
    if tagged:
        return json.loads(tagged.group(1))
    match = re.fullmatch(r"```(?:json)?[ \t]*\r?\n(.*?)\r?\n```", text.strip(), re.DOTALL | re.IGNORECASE)
    return json.loads(match.group(1) if match else text)


def feedback_data(text):
    data = generation_json(text)
    if not isinstance(data, dict) or not isinstance(data.get("errors"), list):
        raise ValueError("反馈必须是包含errors数组的JSON对象")
    return data


def copy_text(text):
    """封装比较只忽略行末空格和CRLF差异，保留措辞、标点与分行。"""
    return '\n'.join(line.rstrip(' \t') for line in text.replace('\r\n', '\n').strip().split('\n'))


def decode_step(step, text, values):
    notes = []
    if step.get("result_json_key"):
        key = step["result_json_key"]
        data = generation_json(text)
        if not isinstance(data, dict) or set(data) != {key}:
            raise ValueError("结果JSON必须恰好包含字段：" + key)
        value = data[key]
        if key == "candidate_id":
            if type(value) is not int:raise ValueError("candidate_id须为整数")
            text = str(value)
        elif key == "score":
            if type(value) not in (int, float):raise ValueError("score须为数字")
            text = str(value)
        else:
            if not isinstance(value, str) or (key != "query" and not value.strip()):
                raise ValueError(key + "须为字符串，非查询结果不能为空")
            text = json.dumps(value, ensure_ascii=False) if key == "query" else value.strip()
    if step.get("output_format") == "translation" and step.get("translation_format") == "json_only":
        data = json.loads(text)
        if not isinstance(data, dict) or not isinstance(data.get("translation"), str) or not data["translation"].strip():
            raise ValueError("译文JSON必须包含非空字符串translation字段")
        text = data["translation"].strip()
    elif step.get("output_format") == "translation" and step.get("translation_format") == "tag":
        blocks = re.findall(r"<translation>(.*?)</translation>", text, re.DOTALL)
        if not blocks:
            blocks = re.findall(r"(?:^|\n)translation>(.*?)</translation>", text, re.DOTALL)
        if len(blocks) != 1 or not blocks[0].strip() or re.search(r"</?(?:translation|notes)\b", blocks[0]):
            raise ValueError("未找到唯一完整且非空的<translation>...</translation>译文")
        if re.search(r"</?think\s*>", blocks[0], re.I):raise ThinkingError("译文标签内部含思考内容，不能作为纯译文")
        if step.get("notes_allowed"):
            note_blocks = re.findall(r"<notes>(.*?)</notes>", text, re.DOTALL)
            if len(note_blocks) > 1:raise ValueError("说明标签重复")
            if note_blocks:notes = [line.strip() for line in note_blocks[0].splitlines() if line.strip()]
        text = blocks[0].strip()
        if step.get("notes_required") and not notes:raise ValueError("此条件要求简短说明")
    elif step.get("output_format") == "translation" and step.get("translation_format") == "plain":
        blocks = re.findall(r"<translation>(.*?)</translation>", text, re.DOTALL)
        text = (blocks[0] if len(blocks) == 1 else text).strip()
        if not text:raise ValueError("中间译文为空")
    elif step.get("output_format") == "translation":
        data = generation_json(text)
        if not isinstance(data, dict) or not isinstance(data.get("translation"), str) or not data["translation"].strip():
            raise ValueError("译文JSON缺少非空translation")
        notes = data.get("notes")
        if not isinstance(notes, list) or any(not isinstance(x, str) for x in notes):
            raise ValueError("notes须为字符串列表")
        if step.get("notes_required") and not notes:
            raise ValueError("此条件要求简短说明")
        text = data["translation"].strip()
    if step.get("output_format") == "score":
        if not re.fullmatch(r"\d{1,3}(?:\.\d)?", text.strip()) or not 0 <= float(text) <= 100:
            raise ValueError("候选评分应为0—100数字")
    if step.get("validate_feedback"):
        source = json.loads(values[step.get("source_ref", "source")])["text"]
        target = values[step["target_ref"]]
        for item in feedback_data(text)["errors"]:
            required = {"summary", "direction", "source_span", "target_span", "category", "severity", "uncertain", "evidence"}
            if not isinstance(item, dict) or not required <= item.keys() or type(item["uncertain"]) is not bool:
                raise ValueError("反馈字段不完整")
            if item["category"] not in ("accuracy", "omission", "terminology", "fluency", "addition", "rhetoric", "tone") or item["severity"] not in ("critical", "major", "minor"):
                raise ValueError("错误类别或严重度非法")
            if any(not isinstance(item[k], str) or not item[k].strip() for k in ("summary", "direction")):
                raise ValueError("反馈概述或方向为空")
            for key, document in [("source_span", source), ("target_span", target)]:
                span = item[key]
                if span is not None:
                    if not isinstance(span, dict) or not {"quote", "start", "end"} <= span.keys():
                        raise ValueError("片段结构错误")
                    a, b, quote = span["start"], span["end"], span["quote"]
                    if type(a) is not int or type(b) is not int or not 0 <= a < b <= len(document) or document[a:b] != quote:
                        raise ValueError("片段引文或Unicode偏移不符")
            evidence = item["evidence"]
            if not isinstance(evidence, dict) or not {"standard", "quote", "reason"} <= evidence.keys():
                raise ValueError("证据字段错误")
            if not isinstance(evidence["quote"], str) or any(not isinstance(evidence[k], str) or not evidence[k].strip() for k in ("standard", "reason")):
                raise ValueError("证据标准及理由须为非空文字，引文须为文字")
            if evidence["quote"] and evidence["quote"] not in source:
                raise ValueError("证据引文不存在")
            if not evidence["quote"] and not item["uncertain"]:
                raise ValueError("缺证据但未保留不确定性")
    if step.get('exact_copy_from') and copy_text(text) != copy_text(values[step['exact_copy_from']]):
        raise ValueError('JSON封装步骤改写了原译文，保留原始返回并记为失败')
    return text, notes


def gate_passes(when, values):
    errors = when.get("errors_nonempty", [])
    errors = [errors] if isinstance(errors, str) else errors
    issues = when.get("issues_present", [])
    issues = [issues] if isinstance(issues, str) else issues
    return all(values[key].strip().rstrip("。.!！") != "无需修改" for key in issues) and all(feedback_data(values[key])["errors"] for key in errors) and (
        "changed" not in when or values[when["changed"][0]] != values[when["changed"][1]])


def local_result(step, values, row):
    op = step["op"]
    if op == "identity":
        return values[step["input"]]
    if op == "random_select":
        choices = step.get("inputs", step.get("select_from"))
        seed = int(digest([row["poem_id"], row["run_id"], len(choices), "random"]), 16)
        return values[random.Random(seed).choice(choices)]
    if op == "argmax":
        scores = [float(values[key]) for key in step["scores"]]
        if any(not 0 <= score <= 100 for score in scores):
            raise ValueError("候选评分越界")
        best = min(range(len(scores)), key=lambda i: (-scores[i], hashlib.sha256(values[step["inputs"][i]].encode()).hexdigest(), step["inputs"][i]))
        return values[step["inputs"][best]]
    if op == "feedback_view":
        fields = {"summary": ["summary", "direction"],
                  "location": ["summary", "direction", "source_span", "target_span"],
                  "evidence": ["summary", "direction", "evidence"],
                  "full": ["summary", "direction", "source_span", "target_span", "evidence"],
                  "tear": ["source_span", "target_span", "category", "severity", "uncertain"],
                  "tear_evidence": ["source_span", "target_span", "category", "severity", "uncertain", "evidence"]}[step["mode"]]
        return json.dumps({"errors": [{k: item[k] for k in dict.fromkeys(fields + ["uncertain"])} for item in feedback_data(values[step["input"]])["errors"]]}, ensure_ascii=False)
    if op == "lookup":
        query = render(step["query"], values).strip()
        if query.rstrip("。.!！") in ("不查询", "无需查询"):query = ""
        try:
            query = json.loads(query)
            if isinstance(query, dict):query = query.get("query", "")
        except json.JSONDecodeError:
            pass
        if not isinstance(query, str):raise ValueError("查询须为字符串")
        excluded = {p["id"] for key in step.get("exclude_from", []) for p in json.loads(values[key])["passages"]}
        corpus = json.loads(values.get("__reference_entries", "[]"))
        ranked = sorted(corpus, key=lambda p: sum(term in query for term in p["terms"]), reverse=True)
        passages, remaining = [], step.get("max_chars", 600)
        for item in ranked:
            if item["id"] in excluded or not query or not any(term in query for term in item["terms"]):continue
            text = item["text"][:remaining]
            if not text:break
            passages.append({"id": item["id"], "text": text, "source_url": item["source_url"]})
            remaining -= len(text)
        return json.dumps({"query": query, "passages": passages}, ensure_ascii=False)
    raise ValueError("未知本地操作：" + op)


def step_refs(step, guards_only=False):
    when = step.get("when", {})
    guards = when.get("errors_nonempty", [])
    refs = set([guards] if isinstance(guards, str) else guards) | set(when.get("changed", []))
    issues = when.get("issues_present", [])
    refs.update([issues] if isinstance(issues, str) else issues)
    if guards_only:return refs
    refs |= set(re.findall(r"\{\{(\w+)\}\}", step.get("prompt", "") + step.get("query", "")))
    for key in ("input", "target_ref", "history_from"):
        if key in step:refs.add(step[key])
    for key in ("inputs", "scores", "select_from", "exclude_from"):
        refs.update(step.get(key, []))
    return refs


def select_result(step, text, values):
    if step.get("select_from"):
        choices = step["select_from"]
        if step.get("ranking"):
            text = re.sub(r"候选\s*", "", text).strip()
            order = re.findall(r"[1-9]", text)
            if re.search(r"[^1-9,，\s]", text) or sorted(map(int, order)) != list(range(1, len(choices) + 1)):
                raise ValueError("排序必须包含每个候选原始ID且恰好一次")
            return values[choices[int(order[0]) - 1]]
        if step.get('selection_format') == 'label':
            match = re.fullmatch(r'候选\s*([1-9]\d*)[。.]?', text.strip())
            if not match:raise ValueError('中间选择必须回答候选N，不接受裸数字')
            text = match.group(1)
        if not re.fullmatch(r"[1-9]\d*", text) or int(text) > len(choices):
            raise ValueError("选择器必须只返回已有候选序号")
        return values[choices[int(text) - 1]]
    return text


def batch_wave(work, model, cache, models=None, sequential=False):
    """只导出输入已齐备的节点；同诗同run同采样槽的相同请求共享。"""
    ready, finished = {}, []
    for row, recipe in work:
        values = dict(row.get("_bindings", {}), source=json.dumps(row["source"], ensure_ascii=False))
        trace, notes, histories, last_translation, failure = [], {}, {}, None, None
        for step in recipe["steps"]:
            if step_refs(step, True) - values.keys():
                if sequential:break
                continue
            if step.get("when") and not gate_passes(step["when"], values):
                if step["fallback"] not in values:continue
                values[step["id"]] = values[step["fallback"]]
                notes[step["id"]] = notes.get(step["fallback"], [])
                histories[step["id"]] = histories.get(step["fallback"], [])
                trace.append({"step": step["id"], "skipped": True, "fallback": step["fallback"]})
                continue
            if step_refs(step) - values.keys():
                if sequential:break
                continue
            if step.get("op", "llm") != "llm":
                values[step["id"]] = local_result(step, values, row)
                notes[step["id"]] = notes.get(step.get("input"), [])
                histories[step["id"]] = histories.get(step.get("input"), [])
                trace.append({"step": step["id"], "op": step["op"], "result": values[step["id"]]})
                continue
            active = models[step["model"]] if step.get("model") else model
            body = stage_body(step, values, active, row["run_id"], histories.get(step.get("history_from"), []))
            key = digest([row["poem_id"], row["run_id"], step.get("sample_slot", step["id"]),
                          active.get("base_url"), body])
            entry = cache.get(key)
            if entry and entry["status"] == "running":
                exc = UnconfirmedCallError("上次调用结果未确认，请先核对台账节点：" + key)
                exc.step_record_id = key
                raise exc
            if entry is None:
                ready[key] = {"id": key, "status": "exported", "request": body, "endpoint": active.get("base_url"),
                              "output_format": step.get("output_format"), "feedback": bool(step.get("validate_feedback")),
                              "selection": bool(step.get("select_from")), "result_json_key": step.get("result_json_key"),
                              "translation_format": step.get("translation_format"), "notes_allowed": bool(step.get("notes_allowed")), "step_id": step["id"]}
            elif entry["status"] in ("error", "ok"):
                try:
                    if entry["status"] == "error":
                        if entry.get("error_kind") == "ThinkingError":raise ThinkingError(entry["error"])
                        raise ValueError("存在失败步骤节点：" + key)
                    visible = response_text(entry["response"], allow_thinking=body.get("thinking", {}).get("type") == "enabled" or body.get("enable_thinking") is True,
                                            allow_tag_surroundings=step.get("translation_format") == "tag", natural_intermediate=step.get('natural_intermediate', False))
                    text, stage_notes = decode_step(step, visible, values)
                    values[step["id"]] = select_result(step, text, values)
                    histories[step["id"]] = body["messages"] + [{"role": "assistant", "content": text if step.get('native_json_key') == 'feedback' else visible}]
                    notes[step["id"]] = stage_notes
                    trace.append({"step": step["id"], "step_record_id": key})
                    if step.get("output_format") == "translation":last_translation = step["id"]
                except ThinkingError as exc:
                    exc.step_record_id = key
                    raise
                except (ValueError, KeyError, IndexError, TypeError) as exc:
                    exc.step_record_id = key
                    if not recipe.get("technical_fallback") or last_translation is None:raise
                    trace.append({"step": step["id"], "step_record_id": key, "validation_error": str(exc)})
                    failure = str(exc)
                    break
            if sequential and step["id"] not in values:
                break  # 实时流程逐步执行，不提前运行后面的独立分支。
        public = {k: v for k, v in row.items() if not k.startswith("_")}
        if failure or all(step["id"] in values for step in recipe["steps"]):
            final = last_translation if failure else recipe["steps"][-1]["id"]
            result = dict(public, status="ok", translation=values[final], trace=trace,
                          **({"technical_failure": True, "fallback_reason": failure} if failure else {}))
            if recipe.get("conversation_mode"):
                result["formatted_result"] = (json.dumps({"translation": values[final]}, ensure_ascii=False) if recipe.get("final_json_output") else
                                              "<translation>" + values[final] + "</translation>")
            else:result["notes"] = notes.get(final, [])
            finished.append(result)
    return ready, finished


def batch_run(args, config, root, model, work, output, models):
    if model.get("provider") not in ("mimo", "qwen"):
        raise ValueError("文件Batch仅支持MiMo和Qwen")
    ledger = root / config.get("steps_output", "results/steps.{model}.jsonl").format(model=args.model)
    cache = {v["id"]: v for v in read_jsonl(ledger)} if ledger.exists() else {}
    if args.import_batch:
        imported = list(read_jsonl(args.import_batch))
        ids = [v["custom_id"] for v in imported]
        if len(ids) != len(set(ids)) or set(ids) - cache.keys():
            raise ValueError("Batch结果存在未知或重复custom_id，请使用原步骤台账")
        for item in imported:
            key = item["custom_id"]
            if cache[key]["status"] != "exported":
                continue  # 同一结果重复导入不覆盖原始记录。
            entry = dict(cache[key], batch_result=item)
            try:
                if item.get("error") or not item.get("response") or item["response"]["status_code"] != 200:
                    raise ValueError("Batch请求失败")
                entry["response"] = item["response"]["body"]
                response_text(entry["response"])
                entry["status"] = "ok"
            except (ValueError, KeyError, IndexError) as exc:
                entry.update(status="error", error=str(exc), error_kind=type(exc).__name__)
            cache[key] = entry
            append_jsonl(ledger, entry)
    ready, finished = batch_wave(work, model, cache, models)
    done = existing_ids(output)
    for row in finished:
        if row["id"] not in done:
            append_jsonl(output, row)
            done.add(row["id"])
    if args.batch_file:
        physical = models[getattr(args, "batch_provider", None) or args.model]
        ready = {k: v for k, v in ready.items() if v["request"]["model"] == physical["model"] and
                 v["endpoint"] == physical["base_url"]}
        if not ready:
            log_event({"event": "batch_no_ready", "message": "无可导出节点：已完成或尚在等待已导出结果"})
            return
        path = Path(args.batch_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8") as file:
            # 先记待处理台账，导出中断时宁可等待核查，不盲目重复付费。
            for entry in ready.values():
                append_jsonl(ledger, dict(entry, batch_file=str(path.resolve())))
            for key, entry in ready.items():
                file.write(json.dumps({"custom_id": key, "method": "POST", "url": "/v1/chat/completions", "body": entry["request"]}, ensure_ascii=False) + "\n")
    log_event({"ready_steps": len(ready), "ready_by_model": {name: sum(v["request"]["model"] == name for v in ready.values()) for name in {v["request"]["model"] for v in ready.values()}},
               "completed_jobs": len(finished), "ledger": str(ledger)})


def cache_usage(raw):
    usage = raw.get("usage") if isinstance(raw, dict) else None
    usage = usage if isinstance(usage, dict) else {}
    details = usage.get("prompt_tokens_details")
    details = details if isinstance(details, dict) else {}
    prompt = usage.get("prompt_tokens")
    cached = details.get("cached_tokens", usage.get("prompt_cache_hit_tokens"))
    prompt = prompt if type(prompt) is int and prompt >= 0 else None
    cached = cached if type(cached) is int and cached >= 0 else None
    rate = f"{cached / prompt:.2%}" if prompt and cached is not None and cached <= prompt else None
    return {"prompt_tokens": prompt, "cached_tokens": cached, "cache_hit_rate": rate}


class RealtimeSteps:
    """进程内共享请求只发送一次；磁盘running与本进程在途请求分开处理。"""
    def __init__(self, cache, ledger, log_api=False):
        self.cache, self.ledger, self.log_api = cache, ledger, log_api
        self.lock, self.inflight, self.totals = threading.RLock(), {}, {}

    def wave(self, row, recipe, model, models):
        with self.lock:
            visible = ChainMap(dict.fromkeys(self.inflight), self.cache)
            return batch_wave([(row, recipe)], model, visible, models, sequential=True)

    def emit(self, entry, row, api_call):
        if not self.log_api:return
        stats = cache_usage(entry.get("response", {})) if api_call else {"prompt_tokens": None, "cached_tokens": None, "cache_hit_rate": None}
        event = {k: row[k] for k in ("generator_track", "condition_id", "poem_id", "run_id") if k in row}
        event.update(event="api_response" if api_call else "step_reused", api_call=api_call,
                     step_id=entry.get("step_id"), model=entry["request"]["model"],
                     status="faild" if entry["status"] == "error" else entry["status"], **stats)
        if api_call:
            totals = self.totals.setdefault((entry["endpoint"], event["model"]), [0, 0, 0])
            if stats["cache_hit_rate"] is not None:
                totals[0] += stats["prompt_tokens"];totals[1] += stats["cached_tokens"];totals[2] += 1
            event["session_cache_hit_rate"] = f"{totals[1] / totals[0]:.2%}" if totals[0] else None
            event["session_cache_reported_calls"] = totals[2]
        if entry.get("error"):event["error"] = entry["error"]
        log_event(event)

    def request(self, key, entry, model, row):
        with self.lock:
            future = self.inflight.get(key)
            if future is None and key in self.cache:
                record = self.cache[key]
                if record["status"] == "running":raise UnconfirmedCallError("上次调用结果未确认，请先核对台账节点：" + key)
                self.emit(record, row, False)
                return record, False
            owner = future is None
            if owner:
                future = self.inflight[key] = Future()
                self.cache[key] = dict(entry, status="running")
                try:append_jsonl(self.ledger, self.cache[key])
                except BaseException as exc:
                    self.inflight.pop(key);future.set_exception(exc);raise
        if not owner:
            record = future.result()
            with self.lock:self.emit(record, row, False)
            return record, False
        try:
            try:record = dict(entry, status="ok", response=chat(model, entry["request"]))
            except Exception as exc:record = dict(entry, status="error", error=str(exc), error_kind=type(exc).__name__, attempts=getattr(exc, 'attempts', []))
            with self.lock:
                append_jsonl(self.ledger, record)
                self.cache[key] = record
                self.inflight.pop(key);future.set_result(record)
                self.emit(record, row, True)
            return record, True
        except BaseException as exc:
            with self.lock:
                self.inflight.pop(key, None)
                if not future.done():future.set_exception(exc)
            raise


def run_recipe(recipe, values, model, models=None, seed=None, trace=None, context=None,
               cache=None, ledger=None, coordinator=None):
    """实时逐步调用；相同请求可从持久步骤台账复用，run和候选槽保持独立。"""
    try:
        source = json.loads(values["source"])
    except json.JSONDecodeError:
        source = {"title": "", "author": "", "text": values["source"]}
    row = dict(context or {}, id="inline", poem_id=(context or {}).get("poem_id", "inline"),
               run_id=seed, source=source, _bindings={k: v for k, v in values.items() if k != "source"})
    cache = cache if cache is not None else {}
    trace = trace if trace is not None else []
    available = dict(models or {}, primary=model)
    while True:
        ready, finished = (coordinator.wave(row, recipe, model, available) if coordinator else
                           batch_wave([(row, recipe)], model, cache, available, sequential=True))
        if finished:
            result = finished[0]
            called = {v["step_record_id"]: v for v in trace if "step_record_id" in v}
            if coordinator:
                with coordinator.lock:
                    for item in result["trace"]:
                        key = item.get("step_record_id")
                        if key and key not in called:coordinator.emit(coordinator.cache[key], row, False)
            trace[:] = [dict(v, **called.get(v.get("step_record_id"), {}),
                            **({"reused": True} if v.get("step_record_id") not in called and "step_record_id" in v else {}))
                        for v in result["trace"]]
            if trace:
                trace[-1]["recipe_result"] = {k: result[k] for k in ("notes", "formatted_result", "technical_failure", "fallback_reason") if k in result}
            return result["translation"], trace
        if not ready:
            raise ValueError("流程无可运行节点，请检查依赖")
        for key, entry in ready.items():
            active = next(m for m in available.values() if m.get("model") == entry["request"]["model"] and m.get("base_url") == entry["endpoint"])
            event = {"step_record_id": key, "request": entry["request"]}
            trace.append(event)
            if coordinator:
                record, called = coordinator.request(key, entry, active, row)
                if called:
                    if "response" in record:event["response"] = record["response"]
                    if "error" in record:event["error"] = record["error"]
                else:event.update(reused=True)
                continue
            cache[key] = dict(entry, status="running")
            if ledger:append_jsonl(ledger, cache[key])
            try:
                event["response"] = chat(active, entry["request"])
                cache[key] = dict(entry, status="ok", response=event["response"])
            except Exception as exc:
                event["error"] = str(exc)
                cache[key] = dict(entry, status="error", error=str(exc), error_kind=type(exc).__name__)
            if ledger:append_jsonl(ledger, cache[key])


def single_turn(recipe):
    return sum(s.get("op", "llm") == "llm" for s in recipe["steps"]) == 1


def repair_recipe(condition, recipe):
    """正常流程与失败重跑共用的修复；G1-C01重采样仅限明确授权重跑。"""
    recipe = copy.deepcopy(recipe)
    # G6-C07沿用冻结配置的反序展示与单候选选择；末轮JSON由通用流程添加。
    # 完整排名不是该因子的要求，不能额外引入排名格式约束。
    if condition == 'G1-C11':
        for step in recipe['steps']:
            if step['id'] == 'feedback1':
                step.update(native_json_key='feedback', result_json_key='feedback')
                step['prompt'] = step['prompt'].replace('只回答“无需修改”；否则用自然语言说明。', '反馈正文写“无需修改”；否则给出具体反馈。')
                step['prompt'] += '\n仅列最重要的至多5项问题，每项最多100字，总共不超过500字。同一原文片段只检查一次，不重复、不逐句复述全文。'
        recipe['technical_fallback'] = False
    elif condition == 'G6-C09':
        for step in recipe['steps']:
            if step['id'] in ('plan1', 'plan2'):step['max_tokens'] = 4096
        recipe['technical_fallback'] = False
    elif condition == 'G1-C01':
        recipe['steps'][0]['sample_slot'] = recipe['steps'][0].get('sample_slot', recipe['steps'][0]['id']) + '_authorized_retry_v1'
    # 其余条件保留原流程，由通用JSON选择与具体失败节点修复处理。
    return recipe


def final_translation_recipe(recipe, condition=None):
    """中间content原样进入后续对话；仅G1-C11解析反馈，末轮提取译文JSON。"""
    recipe = copy.deepcopy(recipe)
    last = recipe['steps'][-1]
    last_selection = bool(last.get('select_from')) and last.get('op', 'llm') == 'llm'
    append_final = last.get('op', 'llm') != 'llm' or bool(last.get('select_from')) or bool(last.get('when'))
    if append_final:last['id'] = 'selected_translation'
    selection_ids = set()
    for step in recipe['steps']:
        if not (condition == 'G1-C11' and step['id'] == 'feedback1'):
            step.pop('native_json_key', None);step.pop('result_json_key', None)
        if step.get('select_from') and step.get('op', 'llm') == 'llm':
            selection_ids.add(step['id'])
            step.pop('select_from', None)
            step.pop('ranking', None)
            step.pop('selection_format', None)
            step.pop('translation_format', None)
    if append_final:
        recipe['steps'].append({'id': 'translation', 'prompt': '将下面已选中的完整英文译文作为最终结果，保留原文措辞和诗行，不添加说明：\n{{selected_translation}}',
            'output_format': 'translation', 'translation_format': 'json_only', 'temperature': 0,
            'max_tokens_by_hanzi': {'70': 2304, '200': 4608, 'default': 9216}})
        if last_selection:
            recipe['steps'][-1].update(history_from='selected_translation',
                prompt='根据上一轮对候选译文的选择，返回所选候选的完整英文译文，保留该候选措辞和诗行，不改写。')
    # 选择结果可以是序号、排序或自然语言；下轮同时看到候选和原始content。
    for step in recipe['steps']:
        if step.get('op', 'llm') != 'llm':continue
        refs = step_refs(step) & selection_ids
        if refs and not step.get('history_from'):step['history_from'] = sorted(refs)[0]
    for step in recipe['steps'][:-1]:
        if step.get('output_format') == 'translation':step['translation_format'] = 'plain'
        if step.get('op', 'llm') == 'llm' and not step.get('native_json_key'):step['natural_intermediate'] = True
    recipe['technical_fallback'] = False
    return recipe


def failure_recipe(condition, recipe, failure, record):
    recipe = final_translation_recipe(repair_recipe(condition, recipe), condition)
    # 多轮可能先后在不同节点截断；下一次补跑不能丢掉已提高的输出上限。
    previous = failure.get('remediation', {}).get('recipe', {})
    prior_limits = {s['id']: s['max_tokens'] for s in previous.get('steps', [])
                    if 'max_tokens' in s and not s.get('max_tokens_by_hanzi')}
    for step in recipe['steps']:
        prior = prior_limits.get(step['id'])
        if prior is not None and prior > step.get('max_tokens', 0):
            step.pop('max_tokens_by_hanzi', None)
            step['max_tokens'] = prior
    step_id = record.get('step_id')
    if step_id == 'translation' and len(recipe['steps']) > 1 and recipe['steps'][-2].get('id') == 'selected_translation':step_id = 'selected_translation'
    for step in recipe['steps']:
        if step['id'] != step_id:continue
        if 'length' in failure.get('error', ''):
            limit = record.get('request', {}).get('max_completion_tokens', record.get('request', {}).get('max_tokens', 2048))
            step.pop('max_tokens_by_hanzi', None)
            step['max_tokens'] = min(32768, max(4096, limit * 2))
    recipe['technical_fallback'] = False
    if 'data_inspection_failed' in json.dumps(record.get('attempts', [])):
        recipe['steps'][0]['sample_slot'] = recipe['steps'][0].get('sample_slot', recipe['steps'][0]['id']) + '_input_rejection_retry_v1'
        recipe['operational_retry_reason'] = 'input_rejection_new_draft; original source and method unchanged'
    return recipe


def normal_flow_work(work):
    """保留逻辑任务ID用于续跑，实际修复协议另行留档，不重跑既有结果。"""
    result = []
    for row, recipe in work:
        fixed = repair_recipe(row['condition_id'], recipe) if row['condition_id'] in ('G6-C07', 'G1-C11', 'G6-C09') else recipe
        fixed = final_translation_recipe(fixed, row['condition_id'])
        if fixed != recipe:
            recipe = fixed
            row = dict(row, generation_protocol={'version': 'normal-flow-v4-selection-content-passthrough',
                'execution_id': digest([row['id'], recipe]), 'recipe': recipe,
                'identity_policy': 'logical ID retained for resume; actual request revision recorded separately'})
        result.append((row, recipe))
    return result


def protocol_issue(row, cache):
    calls = []
    for step in row.get('trace', []):
        record = cache.get(step.get('step_record_id'), {})
        body = step.get('request', record.get('request'))
        raw = step.get('response', record.get('response'))
        if body and raw:calls.append((step.get('step', record.get('step_id')), body, raw))
    if not calls:return 'flow'
    for name, body, _ in calls[:-1]:
        if body.get('response_format') and not (row['condition_id'] == 'G1-C11' and name == 'feedback1'):
            return 'flow'
    if row['condition_id'] == 'G1-C11':
        feedback = next((raw for name, body, raw in calls if name == 'feedback1' and body.get('response_format')), None)
        if not feedback:return 'flow'
        try:plain = json.loads(feedback['choices'][0]['message']['content'])['feedback'].strip()
        except (ValueError, KeyError, IndexError, TypeError):return 'flow'
        assistants = [m['content'] for m in calls[-1][1]['messages'] if m['role'] == 'assistant']
        if not assistants or assistants[-1] != plain:return 'flow'
    _, body, raw = calls[-1]
    try:data = json.loads(raw['choices'][0]['message']['content'])
    except (ValueError, KeyError, IndexError, TypeError):data = None
    if not body.get('response_format') or not isinstance(data, dict) or data.get('translation', '').strip() != row.get('translation', '').strip():
        return 'final_only'
    return None


def format_existing_translation(row):
    """保留已有自然文本流程及选定译文，仅补最后一次模型JSON返回。"""
    row = copy.deepcopy(row)
    row.setdefault('_bindings', {})['existing_translation'] = row['translation']
    return row, {'conversation_mode': True, 'final_json_output': True, 'steps': [
        {'id': 'translation', 'sample_slot': 'protocol_final', 'prompt': '将下面完整英文译文作为最终结果，保留措辞和诗行，不添加说明：\n{{existing_translation}}',
         'output_format': 'translation', 'translation_format': 'json_only', 'temperature': 0,
         'max_tokens_by_hanzi': {'70': 2304, '200': 4608, 'default': 9216}}]}


def conversation_recipe(recipe, final_json=False):
    result = copy.deepcopy(recipe)
    result["conversation_mode"] = True
    if final_json:result["final_json_output"] = True
    for step in result["steps"]:
        step.pop("output_preamble", None)
        if step.get("op", "llm") != "llm":continue
        step.pop("notes_required", None);step.pop("notes_allowed", None)
        if step.get("output_format") == "translation":
            final = step is result["steps"][-1]
            step["translation_format"] = ("json_only" if final_json else "tag") if final else "plain"
            if final and not final_json:step["output_preamble"] = TRANSLATION_PREAMBLE
    return result


def translation_tag_recipe(recipe):
    result = copy.deepcopy(recipe)
    for step in result["steps"]:
        if step.get("op", "llm") != "llm":continue
        step["output_preamble"] = TRANSLATION_PREAMBLE
        if step.get("output_format") != "translation":
            continue
        prompt = step["prompt"]
        step["notes_allowed"] = bool(step.get("notes_required") or "notes中" in prompt or "notes列表" in prompt)
        prompt = re.sub(r"只输出JSON对象[^。\n]*。", "", prompt)
        prompt = prompt.replace("只输出最终JSON", "只输出最终译文标签")
        prompt = prompt.replace("在同一JSON对象中，先完成translation，再在notes列表给", "先完成译文，再在<notes>标签中给")
        prompt = prompt.replace("在notes中", "在<notes>标签中").replace("notes必须为非空列表", "说明标签必须非空")
        prompt = prompt.replace("notes不超过", "说明不超过")
        step.update(prompt=prompt, translation_format="tag", output_preamble=TRANSLATION_PREAMBLE)
    return result


def result_json_recipe(recipe, templates):
    """为所有生成节点给出可填值的统一JSON外壳；原任务要求保留。"""
    result = copy.deepcopy(recipe)
    for step in result["steps"]:
        if step.get("op", "llm") != "llm":continue
        if step.get("translation_format") == "tag":continue
        if step.get("output_format") == "translation":kind = "translation"
        elif step.get("validate_feedback"):kind = "feedback"
        elif step.get("select_from"):kind = "candidate_id"
        elif step.get("output_format") == "score":kind = "score"
        elif step["id"] == "backtranslation":kind = "back_translation"
        elif step["id"].startswith("query"):kind = "query"
        elif step["id"].startswith("plan"):kind = "plan"
        elif step["id"] == "analysis":kind = "analysis"
        else:kind = "result"
        template = json.dumps(templates[kind], ensure_ascii=False, indent=2)
        if kind not in ("translation", "feedback"):step["result_json_key"] = kind
        instruction = ("\n\n统一输出格式：下面模板覆盖前文的外层格式描述；前文要求的内容填写在对应字段值中。"
                       "只替换字段值，保留字段名与数据类型；仅输出一对<result_json>和</result_json>标签及内部合法JSON。"
                       "不要Markdown代码围栏、标签外说明、思考过程或思考标签，英文诗中的换行在JSON字符串中写为\\n。")
        if kind == "translation":instruction += "完整译文填translation；notes按原任务要求填写，未要求说明时保持[]。"
        if kind == "feedback":instruction += "errors模板中的对象展示每项完整字段；发现几项填写几项，无发现时errors=[]，不要保留空白占位项。"
        step["prompt"] += instruction + "\n<result_json>\n" + template + "\n</result_json>"
    return result


def source_first_recipe(recipe):
    """只移动单轮原诗块；任务、示例、术语和输出要求保留。"""
    if not single_turn(recipe):return recipe
    result = copy.deepcopy(recipe)
    step = next(s for s in result["steps"] if s.get("op", "llm") == "llm")
    if step["prompt"].count("{{source}}") != 1:
        raise ValueError("单轮诗歌前缀要求恰好一个{{source}}占位符")
    task = re.sub(r"(?:待译作品数据：\s*|原作：\s*)?\{\{source\}\}", "", step["prompt"], count=1).strip()
    step["prompt"] = "<原诗>\n{{source}}\n</原诗>\n\n<任务>\n" + task + "\n</任务>"
    return result


def schedule_jobs(work):
    """单轮按诗集中；多轮保留条件→诗歌→run的原顺序。"""
    poems, multi = {}, []
    for row, recipe in work:
        if single_turn(recipe):
            poems.setdefault(row["poem_id"], []).append((row, recipe))
        else:
            multi.append((row, recipe))
    return [job for group in poems.values() for job in group] + multi


def run_realtime_work(work, execute, workers):
    """单轮逐诗预热后并发；多轮并发推进完整流程，流程内仍按依赖顺序执行。"""
    if type(workers) is not int or workers < 1:raise ValueError("generation_workers必须是正整数")
    poems, multi = {}, []
    for job in work:
        if single_turn(job[1]):poems.setdefault(job[0]["poem_id"], []).append(job)
        else:multi.append(job)
    def run_parallel(group):
        if workers == 1:
            for job in group:
                if not execute(job):return False
            return True
        with ThreadPoolExecutor(max_workers=workers) as pool:
            remaining = iter(group)
            pending = {pool.submit(execute, job) for job in (next(remaining, None) for _ in range(workers)) if job is not None}
            while pending:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                failure, exception = False, None
                for future in done:
                    try:failure = not future.result() or failure
                    except BaseException as exc:exception = exception or exc
                if failure or exception:
                    for future in pending:future.cancel()
                    if exception:raise exception
                    return False
                for _ in done:
                    job = next(remaining, None)
                    if job is not None:pending.add(pool.submit(execute, job))
        return True

    for group in poems.values():
        if not execute(group[0]):return False
        if not run_parallel(group[1:]):return False
    return run_parallel(multi)


def jobs(config, root, track, selected=None, poem_ids=None):
    repeats = config.get("repeats", 5)
    if type(repeats) is not int or repeats < 1:
        raise ValueError("repeats必须是正整数")
    design = read_json(root / config["design"])
    allowed = {c["id"] for g in design["groups"] for c in g["conditions"]}
    recipes = config["experiments"]
    selected = selected or config["conditions"]
    if len(selected) != len(set(selected)):
        raise ValueError("重复条件")
    poems = read_json(root / config["poems"])["poems"]  # 生成端不读取gold。
    if poem_ids and set(poem_ids) - {p["id"] for p in poems}:
        raise ValueError("未知诗词ID")
    for condition in selected:
        if condition not in allowed or condition not in recipes:
            raise ValueError("条件未启用或未提供实现：" + condition)
        recipe = resolve_recipe(config, condition)
        if config.get("natural_conversations", False):
            recipe = conversation_recipe(recipe, final_json=config.get("final_json_output", False))
        elif config.get("translation_tags", False):
            recipe = translation_tag_recipe(recipe)
        if config.get("output_templates") and not config.get("natural_conversations", False):
            recipe = result_json_recipe(recipe, config["output_templates"])
        if config.get("single_turn_source_first", False):
            recipe = source_first_recipe(recipe)
        known = {"source", "notes_token_limit", "explanation_token_limit", "glossary", "reference", "examples", "__reference_entries"}
        for step in recipe["steps"]:
            if step_refs(step) - known or (step.get("fallback") and step["fallback"] not in known):
                raise ValueError("步骤引用未定义：" + step["id"])
            if step["id"] in known:
                raise ValueError("步骤ID重复或选择了不存在的候选")
            known.add(step["id"])
        if not recipe["steps"]:
            raise ValueError("条件缺少步骤")
        used_models = {track} | {model_name(config, s["model"], track) for s in recipe["steps"] if s.get("model")}
        model_identity = {
            name: {k: v for k, v in model_config(config, name).items()
                   if not k.endswith("_env") and k != "timeout"}
            for name in used_models
        }
        for poem in poems:
            if poem_ids and poem["id"] not in poem_ids:
                continue
            source = {k: poem[k] for k in ("title", "author", "text")}
            bindings = material_values(config, root, recipe, poem)
            for run in range(1, repeats + 1):
                row = {"generator_track": track, "condition_id": condition,
                       "poem_id": poem["id"], "run_id": run, "source": source}
                row["id"] = digest([row, recipe, model_identity, bindings])
                row["_bindings"] = bindings
                yield row, recipe


def _main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="run_config.json")
    parser.add_argument("--model", default="mimo")
    parser.add_argument("--conditions", nargs="+")
    parser.add_argument("--poems", nargs="+")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--workers", type=int, help="最大并发任务数；单轮同诗预热后并发，多轮同时推进完整流程")
    parser.add_argument("--retry-failed", action="store_true", help="按已登记修复方案只重跑失败项，成功后原位替换；原记录自动备份")
    parser.add_argument('--repair-protocol', action='store_true', help='修复失败项及受旧输出协议影响的成功项；合规结果保持')
    parser.add_argument('--task-ids-file', help='只处理JSON数组列出的逻辑任务ID，用于精确恢复当前队列')
    parser.add_argument("--batch-provider", choices=["mimo", "qwen"], help="本次文件的实际模型，跨模型步骤按此分文件")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--batch-file", help="导出本轮已就绪的MiMo或Qwen步骤，不自动上传")
    mode.add_argument("--import-batch", help="导入Batch结果并解锁后续步骤")
    mode.add_argument("--sync", action="store_true", help="仅同步共享台账，推进本主轨结果，不导出或调用接口")
    args = parser.parse_args()
    if args.repair_protocol:args.retry_failed = True
    config, root = load_config(args.config, batch=bool(args.batch_file or args.import_batch or args.sync))
    if args.batch_file or args.import_batch or args.sync:
        config["single_turn_source_first"] = False  # 旧Batch台账保持原请求和节点ID。
        config["translation_tags"] = False
        config["natural_conversations"] = False
        config.pop("output_templates", None)
        config["generation_output"] = config.get("batch_generation_output", config.get("generation_output"))
        for item in config["models"].values():
            item.pop("thinking_enabled", None)
            item.pop("thinking_budget", None)
            item.pop("thinking_completion_allowance", None)
    model = model_config(config, args.model)
    models = {k: model_config(config, k) for k in config["models"]}
    models["partner"] = models[model_name(config, "partner", args.model)]
    work = list(jobs(config, root, args.model, args.conditions, args.poems))
    if args.task_ids_file:
        task_ids = read_json(args.task_ids_file)
        if not isinstance(task_ids, list) or any(not isinstance(key, str) for key in task_ids):raise ValueError('任务ID文件须为字符串数组')
        selected_ids = set(task_ids)
        if selected_ids - {row['id'] for row, _ in work}:raise ValueError('任务ID不属于当前轨道或条件范围')
        work = [(row, rec) for row, rec in work if row['id'] in selected_ids]
    if config.get('natural_conversations', False) and config.get('normal_flow_fixes', True) and not args.retry_failed:
        work = normal_flow_work(work)
    workers = args.workers if args.workers is not None else config.get("generation_workers", 1)
    if type(workers) is not int or workers < 1:raise ValueError("generation_workers必须是正整数")
    # 所有步骤在网络调用前做能力预检。
    for _, recipe in work:
        for step in recipe["steps"]:
            if step.get("op", "llm") == "llm":
                request_body(models[step["model"]] if step.get("model") else model, [], 100)
    output = root / config.get("generation_output", "results/translations.jsonl")
    ledger = root / config.get("realtime_steps_output", "results/realtime_steps.jsonl")
    done = existing_ids(output)
    pending = [(row, rec) for row, rec in work if row["id"] not in done]
    saved_rows = list(read_jsonl(output)) if args.retry_failed and output.exists() else []
    saved_index = {r['id']: i for i, r in enumerate(saved_rows)}
    if args.retry_failed:
        if args.batch_file or args.import_batch or args.sync:raise ValueError('失败修复仅支持实时模式')
        retry_records = load_step_cache(ledger)
        pending = []
        for row, rec in work:
            if row['id'] not in saved_index:continue
            old = saved_rows[saved_index[row['id']]]
            issue = protocol_issue(old, retry_records) if args.repair_protocol and old.get('status') == 'ok' else None
            if old.get('status') != 'faild' and not issue:continue
            record = retry_records.get(old.get('failed_step_record_id'), {})
            raw_content = ((record.get('response', {}).get('choices') or [{}])[0].get('message') or {}).get('content', '')
            try:decoded = json.loads(raw_content)
            except (ValueError, TypeError):decoded = None
            recover_json_string = (old.get('status') == 'faild' and 'translation字段' in old.get('error', '')
                                   and isinstance(decoded, str) and bool(decoded.strip()))
            if recover_json_string:
                row = dict(row, translation=decoded.strip())
                row, rec = format_existing_translation(row)
                rec['reused_translation_record_id'] = old.get('failed_step_record_id')
                row.pop('translation', None)
            elif issue == 'final_only':
                row = dict(row, translation=old['translation'], _prior_trace=old.get('trace', []))
                row, rec = format_existing_translation(row)
                row.pop('translation', None)
            else:rec = failure_recipe(row['condition_id'], rec, old, record)
            pending.append((row, rec))
    if args.dry_run:
        ledger = root / config.get("realtime_steps_output", "results/realtime_steps.jsonl")
        if ledger.exists():
            for _ in read_jsonl(ledger):pass
        log_event({"planned": len(work), "pending": len(pending), "output": str(output), "workers": workers})
        return
    if args.batch_file or args.import_batch or args.sync:
        batch_run(args, config, root, model, pending, output, models)
        return
    if config.get("transport") == "batch":
        raise ValueError("配置要求文件Batch，请指定--batch-file导出；不自动退回实时计费")
    ledger = root / config.get("realtime_steps_output", "results/realtime_steps.jsonl")
    cache = load_step_cache(ledger)
    coordinator = RealtimeSteps(cache, ledger, log_api=True)
    if args.retry_failed:
        if not pending:
            log_event({'event': 'retry_failed_complete', 'pending': 0})
            return
        backup = root / 'results/recovery' / ('failed_retry_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
        backup.mkdir(parents=True)
        for path in (output, ledger):
            if path.exists():shutil.copy2(path, backup / path.name)
        retry_keys = set()
        failed_keys = set()
        retry_keys.update(key for key, record in cache.items() if record['status'] == 'running')
        for row, _ in pending:
            old = saved_rows[saved_index[row['id']]]
            retry_keys.add(old.get('failed_step_record_id'))
            failed_keys.add(old.get('failed_step_record_id'))
            retry_keys.update(t.get('step_record_id') for t in old.get('trace', []))
        for key in retry_keys:
            record = cache.get(key)
            if record and (record['status'] in ('running', 'error') or key in failed_keys):
                append_jsonl(ledger, dict(record, status='retry_authorized', prior_status=record['status'], authorization='explicit --retry-failed', backup=str(backup)))
                cache.pop(key)
        log_event({'event': 'retry_failed_start', 'pending': len(pending), 'backup': str(backup)})
    else:pending = [(row, recipe) for row, recipe in schedule_jobs(work) if row["id"] not in done]
    last_flush = time.monotonic()
    def execute(job):
        nonlocal last_flush
        row, recipe = job
        trace = []
        try:
            text, _ = run_recipe(recipe, dict(row.get("_bindings", {}), source=json.dumps(row["source"], ensure_ascii=False)), model, models, row["run_id"], trace, row,
                                 cache=cache, ledger=ledger, coordinator=coordinator)
            row.update(status="ok", translation=text)
            if trace:row.update(trace[-1].get("recipe_result", {}))
        except Exception as exc:
            key = getattr(exc, "step_record_id", None)
            with coordinator.lock:
                record = coordinator.cache.get(key, {})
                if not record:
                    key = next((t.get("step_record_id") for t in reversed(trace) if t.get("step_record_id")), None)
                    record = coordinator.cache.get(key, {})
                response = record.get("response") or next((t["response"] for t in reversed(trace) if t.get("response")), {})
            choices = response.get("choices") if isinstance(response, dict) else None
            choice = choices[0] if isinstance(choices, list) and choices and isinstance(choices[0], dict) else {}
            message = choice.get("message")
            message = message if isinstance(message, dict) else {}
            row.update(status="faild", error=str(exc), error_kind=type(exc).__name__, content=message.get("content"),
                       reasoning_content=message.get("reasoning_content"), failed_step_record_id=key,
                       attempts=getattr(exc, "attempts", []))
        progress = {k: row[k] for k in ("condition_id", "poem_id", "run_id", "status")}
        if row["status"] == "faild":progress["error"] = row["error"]
        with coordinator.lock:
            result = dict({k: v for k, v in row.items() if not k.startswith("_")}, trace=row.get('_prior_trace', []) + trace)
            if args.retry_failed:
                result['timestamp'] = datetime.now(timezone(timedelta(hours=8))).isoformat(timespec='milliseconds')
                result['remediation'] = {'version': 'repair-v5-selection-content-passthrough', 'original_id': row['id'],
                    'execution_id': digest([row['id'], recipe]), 'recipe': recipe,
                    'prior_error': saved_rows[saved_index[row['id']]].get('error'), 'backup': str(backup)}
                append_jsonl(backup / 'attempts.jsonl', result)
                saved_rows[saved_index[row['id']]] = result
                if time.monotonic() - last_flush >= 3:
                    write_jsonl_atomic(output, saved_rows)
                    last_flush = time.monotonic()
            else:append_jsonl(output, result)
            log_event(progress)
        return True  # 单条失败已落盘，继续其他任务；不重新采样失败任务。
    try:
        if not run_realtime_work(pending, execute, workers):raise SystemExit(1)
    finally:
        if args.retry_failed:write_jsonl_atomic(output, saved_rows)


def main():
    with generation_lock(Path(__file__).resolve().parent / 'results/generation'):
        return _main()


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, KeyError) as exc:
        log_event({"event": "error", "error": str(exc)}, stream=sys.stderr)
        raise SystemExit(1)
