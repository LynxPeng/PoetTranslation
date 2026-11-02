import json
import tempfile
import unittest
import contextlib
import io
from urllib.error import HTTPError
from pathlib import Path
from unittest.mock import patch

import generate
import evaluate


class PipelineTests(unittest.TestCase):
    def test_qwen_explicit_nonthinking(self):
        body = generate.request_body({"provider": "qwen", "model": "qwen-flash"}, [], 64)
        self.assertIs(body["enable_thinking"], False)
        self.assertNotIn("thinking", body)
        self.assertEqual(body["max_tokens"], 64)

    def test_batch_wave_waits_for_dependencies_and_shares_initial(self):
        source = {"title": "诗", "author": "某", "text": "月"}
        work = [({"id": "a", "poem_id": "T01", "run_id": 1, "source": source},
                 {"steps": [{"id": "translation", "sample_slot": "initial", "prompt": "{{source}}"}]}),
                ({"id": "b", "poem_id": "T01", "run_id": 1, "source": source},
                 {"steps": [{"id": "draft", "sample_slot": "initial", "prompt": "{{source}}"},
                            {"id": "translation", "prompt": "改 {{draft}}"}]})]
        ready, finished = generate.batch_wave(work, self.model, {})
        self.assertEqual(len(ready), 1)
        self.assertEqual(finished, [])
        key = next(iter(ready))
        cache = {key: {"id": key, "status": "ok", "response": self.response("initial")}}
        ready, finished = generate.batch_wave(work, self.model, cache)
        self.assertEqual(len(finished), 1)
        self.assertEqual(len(ready), 1)
        self.assertIn("initial", next(iter(ready.values()))["request"]["messages"][0]["content"])

    def setUp(self):
        self.model = {"model": "mimo-v2.6-flash", "base_url": "http://localhost/v1",
                      "api_key_env": "TEST_KEY", "provider": "mimo"}

    def response(self, text="Moonlight", **kwargs):
        return {"choices": [{"message": {"content": text, **kwargs},
                             "finish_reason": "stop"}], "usage": {}}

    def test_non_thinking_and_provider_limit(self):
        body = generate.request_body(self.model, [{"role": "user", "content": "诗"}], 100)
        self.assertEqual(body["thinking"], {"type": "disabled"})
        self.assertEqual(body["max_completion_tokens"], 100)
        self.assertNotIn("reasoning_effort", body)
        with self.assertRaises(ValueError):
            generate.request_body(dict(self.model, extra_body={"thinking": {"type": "enabled"}}), [], 100)

    def test_local_requires_verified_disable_controls(self):
        with self.assertRaises(ValueError):
            generate.request_body({"provider": "local", "model": "qwen"}, [], 100)
        model = {"provider": "local", "model": "qwen", "nonthinking_verified": True,
                 "extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}
        self.assertFalse(generate.request_body(model, [], 100)["chat_template_kwargs"]["enable_thinking"])

    def test_reject_thinking_and_truncation(self):
        for raw in [self.response(reasoning_content="thinking"),
                    dict(self.response(), usage={"completion_tokens_details": {"reasoning_tokens": 1}}),
                    {"choices": [{"message": {"content": "half"}, "finish_reason": "length"}]}]:
            with self.assertRaises(generate.ResponseError):
                generate.response_text(raw)

    def test_score_strict(self):
        self.assertEqual(evaluate.parse_score('{"D1":25,"D2":8,"D3":12,"D4":13,"D5":9,"D6":8.5,"D7":8,"total":83.5}')['total'], 83.5)
        for bad in ["Score: 83", "{\"score\": 83}", "101", "nan", "1e2", "83.55", "-1"]:
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                evaluate.parse_score(bad)

    def test_templates_do_not_reinterpret_inserted_text(self):
        self.assertEqual(generate.render("{{source}} / {{draft}}", {"source": "{{draft}}", "draft": "ok"}), "{{draft}} / ok")
        with self.assertRaises(ValueError):
            generate.render("{{unknown}}", {})

    def test_multistep_and_trace(self):
        recipe = {"steps": [{"id": "draft", "prompt": "{{source}}"},
                            {"id": "final", "prompt": "修订 {{draft}}"}]}
        with patch.object(generate, "chat", side_effect=[self.response("draft"), self.response("final")]) as call:
            text, trace = generate.run_recipe(recipe, {"source": "诗"}, self.model)
        self.assertEqual(text, "final")
        self.assertEqual(len(trace), 2)
        self.assertIn("draft", call.call_args.args[1]["messages"][-1]["content"])

    def test_judge_prefix_and_blinding(self):
        poem = {"id": "T01", "title": "静夜思", "author": "李白", "text": "床前明月光"}
        rubric = {"task": "翻译", "dimensions": [], "internal_anchors": {}, "total_formula": "sum"}
        a = evaluate.judge_messages(poem, rubric, "one", "rules")
        b = evaluate.judge_messages(poem, rubric, "two", "rules")
        self.assertEqual(a[0], b[0])
        self.assertEqual(a[1]["content"].split('"translation":')[0], b[1]["content"].split('"translation":')[0])
        self.assertNotIn("generator_track", json.dumps(a))

    def test_resume_detects_conflicting_rows(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "out.jsonl"
            generate.append_jsonl(path, {"id": "a", "status": "error"})
            self.assertIn("a", generate.existing_ids(path))

    def test_invalid_repeats_fail_before_calls(self):
        config, _ = generate.load_config(Path(__file__).resolve().parents[1] / "run_config.json")
        config["repeats"] = 0
        with self.assertRaises(ValueError):
            list(generate.jobs(config, Path(__file__).resolve().parents[1], "mimo"))

    def test_generation_resume_identity_ignores_unrelated_config(self):
        root = Path(__file__).resolve().parents[1]
        config, _ = generate.load_config(root / "run_config.json")
        config["conditions"] = ["G1-C00"]
        before = [row["id"] for row, _ in generate.jobs(config, root, "mimo", poem_ids=["T01"])]
        config["repeats"] = 6
        config["evaluation_output"] = "different.jsonl"
        config["models"]["deepseek"]["timeout"] = 1
        config["models"]["qwen"]["model"] = "unrelated-other-model"
        after = [row["id"] for row, _ in generate.jobs(config, root, "mimo", poem_ids=["T01"])]
        self.assertEqual(before, after[:5])
        self.assertEqual(len(after), 6)
        config["experiments"]["G1-C00"]["steps"][0]["prompt"] += " Use modern English."
        changed = next(generate.jobs(config, root, "mimo", poem_ids=["T01"]))[0]["id"]
        self.assertNotEqual(before[0], changed)

    def test_candidate_seed_slots_are_distinct(self):
        recipe = {"steps": [{"id": "a", "prompt": "{{source}}"},
                            {"id": "b", "prompt": "{{source}}"},
                            {"id": "final", "prompt": "1={{a}} 2={{b}}", "select_from": ["a", "b"]}]}
        with patch.object(generate, "chat", side_effect=[self.response("first"), self.response("second"), self.response("2")]) as call:
            text, trace = generate.run_recipe(recipe, {"source": "诗"}, dict(self.model, supports_seed=True), seed=101)
        self.assertEqual(text, "second")
        self.assertNotEqual(trace[0]["request"]["seed"], trace[1]["request"]["seed"])

    def fixture_config(self, folder):
        root = Path(__file__).resolve().parents[1]
        config, _ = generate.load_config(root / "run_config.json")
        config.pop("materials", None)
        config.update(translation_tags=False, natural_conversations=False, generation_workers=1)
        for key in ('result_json_templates', 'output_templates', 'batch_generation_output'):
            config.pop(key, None)  # 此夹具专门验证旧协议和Batch兼容路径。
        config["conditions"] = ["G1-C00"]
        draft = {"id": "draft", "prompt": "{{source}}", "sample_slot": "initial"}
        config["experiments"] = {
            "G1-C00": {"steps": [draft]},
            "G1-C06": {"steps": [draft, {"id": "errors", "prompt": "check {{draft}}"},
                                   {"id": "translation", "prompt": "revise {{draft}} {{errors}}"}]}}
        for key in ("poems", "design", "gold", "manifest", "rubric", "judge_prompt"):
            config[key] = str(root / config[key])
        config["repeats"] = 2
        config["transport"] = "realtime"
        config["generation_output"] = str(Path(folder) / "translations.jsonl")
        config["evaluation_output"] = str(Path(folder) / "scores.jsonl")
        path = Path(folder) / "config.json"
        path.write_text(json.dumps(config, ensure_ascii=False), encoding="utf-8")
        return path, config

    def test_end_to_end_and_resume(self):
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
            path, config = self.fixture_config(folder)
            with patch("sys.argv", ["generate.py", "--config", str(path), "--poems", "T01"]), patch.object(generate, "chat", return_value=self.response("Moonlight at my bedside")) as caller:
                generate.main()
                generate.main()
                self.assertEqual(caller.call_count, 2)
                self.assertNotIn("key_annotations", json.dumps(caller.call_args.args[1]))
            packet = '{"D1":25,"D2":8,"D3":12,"D4":13,"D5":9,"D6":8.5,"D7":8,"total":83.5}'
            with patch("sys.argv", ["evaluate.py", "--config", str(path)]), patch.object(evaluate, "chat", return_value=self.response(packet)) as caller, \
                    patch.object(evaluate, 'read_json', wraps=generate.read_json) as reads:
                with patch("sys.argv", ["evaluate.py", "--config", str(path), "--dry-run"]):
                    evaluate.main()
                    caller.assert_not_called()
                evaluate.main()
                evaluate.main()
                self.assertEqual(caller.call_count, 1)
                body = caller.call_args.args[1]
                self.assertNotIn("generator_track", json.dumps(body))
                self.assertEqual(body["max_tokens"], 512)
                self.assertEqual(body['response_format'], {'type': 'json_object'})
                self.assertEqual(body['thinking'], {'type': 'disabled'})
                self.assertEqual(set(json.loads(body['messages'][1]['content'])), {'source', 'translation'})
                self.assertFalse({'gold_annotations.json', 'manifest.json'} & {Path(call.args[0]).name for call in reads.call_args_list})
            self.assertEqual([x["score"] for x in generate.read_jsonl(config["evaluation_output"])], [83.5, 83.5])
            self.assertTrue(all(r['D1'] == 25 and r['total'] == 83.5 for r in generate.read_jsonl(config['evaluation_output'])))

    def test_failed_generation_prints_error_detail(self):
        with tempfile.TemporaryDirectory() as folder:
            path, config = self.fixture_config(folder)
            config['experiments']['G1-C00']['steps'][0]['output_format'] = 'translation'
            config['repeats'] = 3
            path.write_text(json.dumps(config), encoding='utf-8')
            out = io.StringIO()
            with contextlib.redirect_stdout(out), patch('sys.argv', ['generate.py', '--config', str(path), '--poems', 'T01']), \
                    patch.object(generate, 'chat', side_effect=[{'choices': [None]}, self.response('not JSON'), self.response('{"translation":"Moon","notes":[]}')]) as caller:
                generate.main()
                generate.main()
            progress = [json.loads(line) for line in out.getvalue().splitlines()]
            failed = next(row for row in progress if row['status'] == 'faild')
            self.assertTrue(failed.get('error'))
            records = list(generate.read_jsonl(config['generation_output']))
            self.assertEqual([r['status'] for r in records], ['faild', 'faild', 'ok'])
            self.assertIsNone(records[0]['content'])
            self.assertEqual(records[1]['content'], 'not JSON')
            self.assertEqual(caller.call_count, 3)

    def test_batch_roundtrip_without_network(self):
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
            path, config = self.fixture_config(folder)
            batch = Path(folder) / "requests.jsonl"
            results = Path(folder) / "results.jsonl"
            with patch("sys.argv", ["generate.py", "--config", str(path), "--poems", "T01", "--batch-file", str(batch)]), patch.object(generate, "chat") as caller:
                generate.main()
                caller.assert_not_called()
            for row in generate.read_jsonl(batch):
                generate.append_jsonl(results, {"custom_id": row["custom_id"], "response": {"status_code": 200, "body": self.response("Batch translation")}})
            with patch("sys.argv", ["generate.py", "--config", str(path), "--poems", "T01", "--import-batch", str(results)]), patch.object(generate, "chat") as caller:
                generate.main()
                caller.assert_not_called()
            self.assertEqual(len(list(generate.read_jsonl(config["generation_output"]))), 2)

    def test_multistep_batch_cli_both_providers_no_reexport(self):
        for track in ("mimo", "qwen"):
            with self.subTest(track=track), tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
                path, config = self.fixture_config(folder)
                config["conditions"] = ["G1-C00", "G1-C06"]
                path.write_text(json.dumps(config, ensure_ascii=False), encoding="utf-8")
                base = ["generate.py", "--config", str(path), "--model", track, "--poems", "T01"]
                for wave in range(3):
                    batch = Path(folder) / f"wave{wave}.jsonl"
                    with patch("sys.argv", base + ["--batch-file", str(batch)]):
                        generate.main()
                    requests = list(generate.read_jsonl(batch))
                    self.assertEqual(len(requests), 2)
                    again = Path(folder) / f"duplicate{wave}.jsonl"
                    with patch("sys.argv", base + ["--batch-file", str(again)]):
                        generate.main()
                    self.assertFalse(again.exists())
                    results = Path(folder) / f"out{wave}.jsonl"
                    for item in requests:
                        if track == "qwen":self.assertIs(item["body"]["enable_thinking"], False)
                        generate.append_jsonl(results, {"custom_id": item["custom_id"], "response": {"status_code": 200, "body": self.response("[]" if wave == 1 else "translation")}})
                    with patch("sys.argv", base + ["--import-batch", str(results)]):
                        generate.main()
                self.assertEqual(len(list(generate.read_jsonl(config["generation_output"]))), 4)

    def test_transport_retry_keeps_disabled_payload(self):
        response = io.BytesIO(json.dumps(self.response()).encode())
        body = generate.request_body(self.model, [{"role": "user", "content": "诗"}], 100)
        with patch.dict("os.environ", {"TEST_KEY": "fake-test-key"}), patch.object(generate.request, "urlopen", side_effect=[HTTPError("url", 429, "busy", {}, None), response]) as caller, patch.object(generate.time, "sleep"), patch.object(generate, 'wait_for_rate_slot'):
            raw = generate.chat(self.model, body)
        self.assertEqual(len(raw["_attempts"]), 2)
        self.assertEqual(json.loads(caller.call_args.args[0].data)["thinking"], {"type": "disabled"})

    def test_bad_judge_answer_recorded_without_fake_score(self):
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
            path, config = self.fixture_config(folder)
            with patch("sys.argv", ["generate.py", "--config", str(path), "--poems", "T01"]), patch.object(generate, "chat", return_value=self.response("Moonlight")):
                generate.main()
            with patch("sys.argv", ["evaluate.py", "--config", str(path)]), patch.object(evaluate, "chat", return_value=self.response("Score: 80")) as caller:
                evaluate.main()
                self.assertEqual(caller.call_count, 1)
            row = next(generate.read_jsonl(config["evaluation_output"]))
            self.assertEqual(row["status"], "error")
            self.assertNotIn("score", row)
            self.assertEqual(row["response"]["choices"][0]["message"]["content"], "Score: 80")
            with patch("sys.argv", ["evaluate.py", "--config", str(path)]), patch.object(evaluate, "chat", return_value=self.response("80")) as caller:
                evaluate.main()
                caller.assert_not_called()


if __name__ == "__main__":
    unittest.main()
