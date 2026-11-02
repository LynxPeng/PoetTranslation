import json
import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import generate


class ResultProtocolTests(unittest.TestCase):
    def test_tagged_json_and_no_ambiguous_blocks(self):
        payload = '{"translation":"Moon","notes":[]}'
        self.assertEqual(generate.generation_json('<result_json>\n' + payload + '\n</result_json>'), json.loads(payload))
        for text in ('Before<result_json>' + payload + '</result_json>',
                     '<result_json>' + payload + '</result_json>After',
                     '<result_json>{}</result_json><result_json>{}</result_json>',
                     '<result_json>{}<result_json>'):
            with self.assertRaises(ValueError):generate.generation_json(text)

    def test_generation_thinking_on_judge_stays_off(self):
        mimo = {'provider': 'mimo', 'model': 'mimo-v2.6-flash', 'thinking_enabled': True}
        body = generate.request_body(mimo, [], 2304)
        self.assertEqual(body['thinking'], {'type': 'enabled'})
        self.assertNotIn('temperature', body)
        self.assertNotIn('top_p', body)
        self.assertNotIn('reasoning_effort', body)
        qwen = {'provider': 'qwen', 'model': 'qwen-flash', 'thinking_enabled': True, 'thinking_budget': 1}
        body = generate.request_body(qwen, [], 2304)
        self.assertIs(body['enable_thinking'], True)
        self.assertEqual(body['thinking_budget'], 1)
        judge = generate.request_body({'provider': 'deepseek', 'model': 'deepseek-flash'}, [], 64, temperature=0)
        self.assertEqual(judge['thinking'], {'type': 'disabled'})
        with self.assertRaises(ValueError):generate.request_body(dict(mimo, thinking_budget=1), [], 64)
        with self.assertRaises(ValueError):generate.request_body(dict(qwen, thinking_budget=0), [], 64)

    def test_reasoning_is_separate_from_answer(self):
        raw = {'choices': [{'message': {'content': '<result_json>{"translation":"Moon","notes":[]}</result_json>',
                                        'reasoning_content': 'internal'}, 'finish_reason': 'stop'}],
               'usage': {'completion_tokens_details': {'reasoning_tokens': 12}}}
        self.assertEqual(generate.generation_json(generate.response_text(raw, allow_thinking=True))['translation'], 'Moon')
        with self.assertRaises(generate.ThinkingError):generate.response_text(raw)
        raw['choices'][0]['message']['content'] = '<think>leaked</think><result_json>{}</result_json>'
        with self.assertRaises(generate.ThinkingError):generate.response_text(raw, allow_thinking=True)

    def test_short_mimo_stages_have_total_output_headroom(self):
        model = {'provider': 'mimo', 'model': 'mimo-v2.6-flash', 'thinking_enabled': True,
                 'thinking_completion_allowance': 1024}
        body = generate.stage_body({'id': 'score', 'prompt': 'Score', 'max_tokens': 64}, {}, model, 1)
        self.assertEqual(body['max_completion_tokens'], 1088)
        judge = generate.request_body({'provider': 'deepseek', 'model': 'deepseek-flash'}, [], 64, temperature=0)
        self.assertEqual(judge['max_tokens'], 64)

    def test_wrapped_scalar_results_are_typed(self):
        for step, data, expected in (
            ({'result_json_key': 'candidate_id', 'select_from': ['a', 'b']}, {'candidate_id': 2}, '2'),
            ({'result_json_key': 'score', 'output_format': 'score'}, {'score': 80}, '80'),
            ({'result_json_key': 'analysis'}, {'analysis': '关系与依据'}, '关系与依据'),
            ({'result_json_key': 'query'}, {'query': ''}, '""')):
            text, _ = generate.decode_step(step, '<result_json>' + json.dumps(data) + '</result_json>', {})
            self.assertEqual(text, expected)
        for value in (True, '1', 1.5):
            with self.assertRaises(ValueError):
                generate.decode_step({'result_json_key': 'candidate_id', 'select_from': ['a']}, json.dumps({'candidate_id': value}), {})

    def test_template_has_fields_and_does_not_mutate_original(self):
        recipe = {'steps': [{'id': 'draft', 'prompt': 'Task {{source}}', 'output_format': 'translation'}]}
        templates = {'translation': {'translation': '', 'notes': []}}
        prepared = generate.result_json_recipe(recipe, templates)
        self.assertNotIn('<result_json>', recipe['steps'][0]['prompt'])
        prompt = prepared['steps'][0]['prompt']
        self.assertIn('<result_json>', prompt)
        self.assertIn('"translation": ""', prompt)
        self.assertIn('"notes": []', prompt)
        self.assertTrue(prompt.endswith('</result_json>'))

    def test_current_config_cli_thinking_and_tagged_results_resume(self):
        root = Path(__file__).resolve().parents[1]
        config, _ = generate.load_config(root / 'run_config.json')
        config['models']['mimo']['thinking_enabled'] = True  # 此用例单独验证历史思考模式兼容。
        config['result_json_templates'] = str(root / 'configs/result_json_templates.json')
        for key in ('poems', 'design', 'result_json_templates'):config[key] = str(root / config[key])
        config.update(conditions=['G1-C00'], repeats=2, translation_tags=False, natural_conversations=False, generation_workers=1)
        config.pop('materials')
        raw = {'choices': [{'message': {'content': '<result_json>{"translation":"Moon","notes":[]}</result_json>',
                                        'reasoning_content': 'mock internal reasoning'}, 'finish_reason': 'stop'}],
               'usage': {'completion_tokens_details': {'reasoning_tokens': 1}}}
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
            config['generation_output'] = str(Path(folder) / 'translations.jsonl')
            config['realtime_steps_output'] = str(Path(folder) / 'steps.jsonl')
            path = Path(folder) / 'config.json'
            path.write_text(json.dumps(config), encoding='utf-8')
            with patch('sys.argv', ['generate.py', '--config', str(path), '--poems', 'T01']), \
                    patch.object(generate, 'chat', return_value=raw) as caller:
                generate.main()
                generate.main()
                self.assertEqual(caller.call_count, 2)
                self.assertEqual(caller.call_args.args[1]['thinking'], {'type': 'enabled'})
            rows = list(generate.read_jsonl(config['generation_output']))
            self.assertEqual([r['translation'] for r in rows], ['Moon', 'Moon'])
            self.assertNotIn('mock internal reasoning', ''.join(r['translation'] for r in rows))


if __name__ == '__main__':unittest.main()
