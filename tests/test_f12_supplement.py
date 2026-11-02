import unittest
import contextlib
import io
import json
from pathlib import Path
import tempfile
from unittest.mock import patch
import generate


class F12SupplementTests(unittest.TestCase):
    def test_isolated_merge_is_idempotent_and_preserves_success(self):
        import f12_isolation
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
            root = Path(folder)
            (root / 'run_config.json').write_text(json.dumps({'generation_output': 'results/main.jsonl'}), encoding='utf-8')
            original = {'id': 'base', 'status': 'ok', 'translation': 'Original'}
            generate.append_jsonl(root / 'results/main.jsonl', original)
            generate.append_jsonl(root / 'results/main.jsonl', {'id': 'failed', 'status': 'faild'})
            base_before = next(generate.read_jsonl(root / 'results/main.jsonl'))
            for row in ({'id': 'base', 'status': 'ok', 'translation': 'Must not replace'},
                        {'id': 'failed', 'status': 'ok', 'translation': 'Recovered'},
                        {'id': 'new', 'status': 'ok', 'translation': 'New'}):
                generate.append_jsonl(root / 'results/translations.f12_qwen.jsonl', row)
            generate.append_jsonl(root / 'results/realtime_steps.f12.jsonl', {'id': 'old', 'status': 'ok'})
            old = next(generate.read_jsonl(root / 'results/realtime_steps.f12.jsonl'))
            generate.append_jsonl(root / 'results/realtime_steps.f12_qwen.jsonl', old)
            generate.append_jsonl(root / 'results/realtime_steps.f12_qwen.jsonl', {'id': 'new', 'status': 'ok'})
            with patch.object(f12_isolation, 'ROOT', root):
                f12_isolation.merge_qwen()
                f12_isolation.merge_qwen()
            rows = list(generate.read_jsonl(root / 'results/main.jsonl'))
            self.assertEqual(len(rows), 3)
            self.assertEqual(rows[0], base_before)
            self.assertTrue(all(row['status'] == 'ok' for row in rows))
            self.assertEqual(len(list(generate.read_jsonl(root / 'results/realtime_steps.f12.jsonl'))), 2)

    def test_exact_copy_request_supplies_literal_json(self):
        step = {'id': 'translation', 'prompt': '{{draft}}', 'translation_format': 'json_only', 'exact_copy_from': 'draft'}
        body = generate.stage_body(step, {'draft': 'Moon\nHome'}, {'provider': 'qwen', 'model': 'qwen-flash'}, 1)
        self.assertIn('{"translation": "Moon\\nHome"}', body['messages'][-1]['content'])

    def test_formatting_step_cannot_rewrite_translation(self):
        step = {'output_format': 'translation', 'translation_format': 'json_only', 'exact_copy_from': 'draft'}
        self.assertEqual(generate.decode_step(step, '{"translation":"Moon\\nHome"}', {'draft': 'Moon\nHome'})[0], 'Moon\nHome')
        self.assertEqual(generate.decode_step(step, '{"translation":"Moon\\nHome"}', {'draft': 'Moon  \nHome'})[0], 'Moon\nHome')
        with self.assertRaisesRegex(ValueError, '改写'):
            generate.decode_step(step, '{"translation":"Changed"}', {'draft': 'Moon\nHome'})

    def test_supported_tracks_and_budgets_are_explicit(self):
        from f12_supplement import build_config
        for track, count in (('mimo', 4), ('qwen', 6)):
            config = build_config(track)
            self.assertEqual(len(config['conditions']), count)
            self.assertEqual(config['repeats'], 5)
            self.assertEqual(config['generation_output'], 'results/translations.json_mode_no_thinking.jsonl')
            for condition in config['conditions']:
                recipe = generate.conversation_recipe(config['experiments'][condition], final_json=True)
                for step in recipe['steps']:
                    model = config['models'][step['model']]
                    values = {'source': '{}', 'draft': 'Moon'}
                    body = generate.stage_body(step, values, model, 1)
                    enabled = condition not in ('G2-C09', 'G2-C10') and step['id'] != 'translation'
                    actual = body.get('enable_thinking', body.get('thinking', {}).get('type') == 'enabled')
                    self.assertEqual(actual, enabled)
                    if track == 'qwen' and enabled:
                        self.assertEqual(body['thinking_budget'], 16384 if condition in ('G2-C06', 'G2-C08') else 4096)
                    if step['id'] == 'translation':
                        self.assertEqual(body['response_format'], {'type': 'json_object'})
                        self.assertEqual(step['exact_copy_from'], 'draft')
