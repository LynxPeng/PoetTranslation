import unittest
import io
import json
from urllib.error import HTTPError
from unittest.mock import patch
import generate


class MimoRecoveryTests(unittest.TestCase):
    def test_actual_g1c11_is_identical_for_both_tracks_except_model_roles(self):
        config, root = generate.load_config('run_config.json')
        for track in ('mimo', 'qwen'):
            with self.subTest(track=track):
                job = next(generate.jobs(config, root, track, ['G1-C11'], ['T01']))
                row, recipe = generate.normal_flow_work([job])[0]
                models = {name: generate.model_config(config, name) for name in config['models']}
                models['partner'] = models[generate.model_name(config, 'partner', track)]
                def raw(text):return {'choices': [{'message': {'content': text}, 'finish_reason': 'stop'}]}
                with patch.object(generate, 'chat', side_effect=[raw('Moon.'), raw('{"feedback":"保留月光"}'), raw('{"translation":"Moonlight."}')]) as caller:
                    result, _ = generate.run_recipe(recipe, {'source': '{"text":"月"}'}, models[track], models, seed=1)
                self.assertEqual(result, 'Moonlight.')
                self.assertEqual(caller.call_args_list[1].args[0]['model'], models['partner']['model'])
                self.assertEqual(caller.call_args_list[1].args[1]['response_format'], {'type': 'json_object'})
                self.assertEqual([m['content'] for m in caller.call_args.args[1]['messages'] if m['role'] == 'assistant'][-1], '保留月光')

    def test_final_format_does_not_inherit_feedback_system_message(self):
        history = [{'role': 'system', 'content': 'Return JSON only: {"feedback":"check"}'},
                   {'role': 'user', 'content': 'Check'}, {'role': 'assistant', 'content': '{"feedback":"revise"}'}]
        body = generate.stage_body({'id': 'final', 'prompt': 'Translate', 'translation_format': 'json_only'}, {},
            {'provider': 'mimo', 'model': 'mimo-v2.6-flash'}, 1, history)
        systems = [m for m in body['messages'] if m['role'] == 'system']
        self.assertEqual(systems, [generate.TRANSLATION_JSON_MESSAGE])
        self.assertEqual(len(body['messages']), 4)

    def test_all_current_normal_flows_end_in_translation_json(self):
        config, root = generate.load_config('run_config.json')
        work = generate.normal_flow_work(list(generate.jobs(config, root, 'mimo', poem_ids=['T01'])))
        for row, recipe in work:
            with self.subTest(condition=row['condition_id']):
                last = recipe['steps'][-1]
                self.assertEqual(last.get('op', 'llm'), 'llm')
                self.assertEqual(last.get('output_format'), 'translation')
                self.assertEqual(last.get('translation_format'), 'json_only')
                for step in recipe['steps']:
                    if step.get('select_from'):
                        self.assertNotIn('native_json_key', step)
                    if step.get('native_json_key'):
                        self.assertEqual(row['condition_id'], 'G1-C11')
                        self.assertEqual(step['native_json_key'], 'feedback')
                for step in recipe['steps'][:-1]:
                    self.assertNotEqual(step.get('translation_format'), 'json_only')

    def test_g1c11_extracts_json_feedback_before_next_turn(self):
        recipe = {'conversation_mode': True, 'final_json_output': True, 'steps': [
            {'id': 'draft', 'prompt': '{{source}}', 'output_format': 'translation', 'translation_format': 'plain'},
            {'id': 'feedback1', 'prompt': 'Check', 'history_from': 'draft', 'native_json_key': 'feedback', 'result_json_key': 'feedback'},
            {'id': 'translation', 'prompt': 'Revise', 'history_from': 'feedback1', 'output_format': 'translation', 'translation_format': 'json_only'}]}
        def raw(text):return {'choices': [{'message': {'content': text}, 'finish_reason': 'stop'}]}
        with patch.object(generate, 'chat', side_effect=[raw('Moon.'), raw('{"feedback":"请保持月亮意象"}'), raw('{"translation":"Moonlight."}')]) as caller:
            result, trace = generate.run_recipe(recipe, {'source': '{"text":"月"}'}, {'provider': 'mimo', 'model': 'mimo-v2.6-flash', 'base_url': 'test'}, seed=1)
        self.assertEqual(result, 'Moonlight.')
        last = caller.call_args.args[1]
        self.assertEqual([m['content'] for m in last['messages'] if m['role'] == 'assistant'][-1], '请保持月亮意象')
        self.assertEqual([m for m in last['messages'] if m['role'] == 'system'], [generate.TRANSLATION_JSON_MESSAGE])
        self.assertIsNone(generate.protocol_issue({'condition_id': 'G1-C11', 'translation': result, 'trace': trace}, {}))

    def test_429_keeps_server_detail_and_shared_backoff(self):
        model = {'provider': 'mimo', 'model': 'mimo-v2.6-flash', 'base_url': 'http://local', 'api_key_env': 'RATE_TEST'}
        error = HTTPError('url', 429, 'busy', {'Retry-After': '12'}, io.BytesIO(b'{"error":{"message":"rate limited secret"}}'))
        response = io.BytesIO(json.dumps({'choices': []}).encode())
        with patch.dict('os.environ', {'RATE_TEST': 'secret'}), patch.object(generate, 'wait_for_rate_slot'), \
             patch.object(generate, 'rate_cooldown') as cooldown, patch.object(generate.random, 'uniform', return_value=0), \
             patch.object(generate.request, 'urlopen', side_effect=[error, response]):
            result = generate.chat(model, {})
        self.assertEqual(result['_attempts'][0]['http_status'], 429)
        self.assertIn('rate limited [redacted]', result['_attempts'][0]['server_error'])
        cooldown.assert_called_once_with(model, 12)

    def test_translation_extra_fields_and_natural_selection(self):
        self.assertEqual(generate.decode_step({'output_format': 'translation', 'translation_format': 'json_only'},
            '{"translation":"Moon","explanation":"extra"}', {})[0], 'Moon')
        step = {'select_from': ['a', 'b'], 'selection_format': 'label'}
        self.assertEqual(generate.select_result(step, '候选2', {'a': 'A', 'b': 'B'}), 'B')
        with self.assertRaises(ValueError):generate.select_result(step, '2', {'a': 'A', 'b': 'B'})
        with self.assertRaises(ValueError):
            generate.select_result({'select_from': ['a', 'b']}, '候选1，候选2', {'a': 'A', 'b': 'B'})

    def test_rate_gate_spaces_requests_and_honors_shared_cooldown(self):
        now = [100.0]
        model = {'provider': 'mimo', 'model': 'test-model', 'base_url': 'test', 'api_key_env': 'TEST', 'requests_per_minute': 80}
        with patch.object(generate.time, 'monotonic', side_effect=lambda: now[0]), \
             patch.object(generate.time, 'sleep', side_effect=lambda delay: now.__setitem__(0, now[0] + delay)):
            generate.wait_for_rate_slot(model)
            generate.wait_for_rate_slot(model)
            self.assertAlmostEqual(now[0], 100.75)
            generate.rate_cooldown(model, 10)
            generate.wait_for_rate_slot(model)
            self.assertGreaterEqual(now[0], 110.75)

    def test_default_has_no_proactive_rpm_limit(self):
        with patch.object(generate.time, 'sleep') as sleep:
            for _ in range(100):generate.wait_for_rate_slot({'provider': 'mimo', 'model': 'mimo-v2.6-flash'})
        sleep.assert_not_called()

    def test_recovery_structures_only_failed_thinking_step_and_raises_length_limit(self):
        recipe = {'steps': [{'id': 'draft', 'prompt': 'Translate', 'output_format': 'translation'},
                            {'id': 'feedback1', 'prompt': 'Check', 'output_format': 'feedback_text', 'max_tokens': 1024},
                            {'id': 'translation', 'prompt': 'Revise', 'output_format': 'translation', 'translation_format': 'json_only'}]}
        failure = {'error_kind': 'ThinkingError', 'error': 'think tag'}
        fixed = generate.failure_recipe('G1-C06', recipe, failure, {'step_id': 'feedback1'})
        self.assertNotIn('native_json_key', fixed['steps'][1])
        self.assertTrue(fixed['steps'][1]['natural_intermediate'])
        self.assertEqual(fixed['steps'][0]['prompt'], recipe['steps'][0]['prompt'])
        fixed = generate.failure_recipe('G1-C06', recipe, {'error': '接口未正常完成：length'}, {'step_id': 'feedback1', 'request': {'max_completion_tokens': 1024}})
        self.assertGreaterEqual(fixed['steps'][1]['max_tokens'], 4096)

    def test_repeated_failure_keeps_previously_increased_step_limits(self):
        recipe = {'steps': [{'id': 'draft', 'prompt': 'Translate'},
                            {'id': 'feedback1', 'prompt': 'Check', 'max_tokens': 1024},
                            {'id': 'feedback2', 'prompt': 'Check again', 'max_tokens': 1024},
                            {'id': 'translation', 'prompt': 'Revise', 'output_format': 'translation', 'translation_format': 'json_only'}]}
        first = generate.failure_recipe('G1-C10', recipe, {'error': 'length'},
            {'step_id': 'feedback1', 'request': {'max_tokens': 1024}})
        failure = {'error': 'length', 'remediation': {
            'version': 'repair-v5-selection-content-passthrough', 'recipe': first}}
        second = generate.failure_recipe('G1-C10', recipe, failure,
            {'step_id': 'feedback2', 'request': {'max_tokens': 1024}})
        self.assertEqual(second['steps'][1]['max_tokens'], 4096)
        self.assertEqual(second['steps'][2]['max_tokens'], 4096)

