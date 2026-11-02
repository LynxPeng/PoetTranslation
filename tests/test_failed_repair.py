import unittest
import contextlib
import io
import json
import tempfile
from pathlib import Path
from unittest.mock import patch
import generate


class FailedRepairTests(unittest.TestCase):
    def test_failed_json_string_only_gets_fresh_final_format_request(self):
        config, root = generate.load_config('run_config.json')
        config.update(conditions=['G2-C01'], repeats=1)
        for field in ('poems', 'design'):config[field] = str(root / config[field])
        config.pop('materials', None)
        def raw(text):return {'choices': [{'message': {'content': text}, 'finish_reason': 'stop'}]}
        with tempfile.TemporaryDirectory() as folder:
            config['generation_output'] = str(Path(folder) / 'translations.jsonl')
            config['realtime_steps_output'] = str(Path(folder) / 'steps.jsonl')
            path = Path(folder) / 'config.json';path.write_text(json.dumps(config), encoding='utf-8')
            args = ['generate.py', '--config', str(path), '--model', 'qwen', '--poems', 'T01']
            with contextlib.redirect_stdout(io.StringIO()), patch.object(generate, 'chat', side_effect=[raw('"Moon."'), raw('{"translation":"Moon."}')]) as caller:
                with patch('sys.argv', args):generate.main()
                with patch('sys.argv', args + ['--retry-failed']):generate.main()
                self.assertEqual(caller.call_count, 2)
                self.assertIn('Moon.', caller.call_args.args[1]['messages'][-1]['content'])
            result = next(generate.read_jsonl(config['generation_output']))
            self.assertEqual(result['status'], 'ok')
            self.assertIn('reused_translation_record_id', result['remediation']['recipe'])

    def test_persistent_windows_rename_lock_uses_backed_up_inplace_write(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'results.jsonl'
            path.write_text('{"old":true}\n', encoding='utf-8')
            with patch.object(Path, 'replace', side_effect=PermissionError('viewer denies delete sharing')), patch.object(generate.time, 'sleep'):
                generate.write_jsonl_atomic(path, [{'new': True}])
            self.assertEqual(list(generate.read_jsonl(path)), [{'new': True}])

    def test_normal_cli_passes_selection_content_then_final_json_and_resumes(self):
        config, root = generate.load_config('run_config.json')
        config.update(conditions=['G6-C07'], repeats=1)
        for field in ('poems', 'design'):config[field] = str(root / config[field])
        config.pop('materials', None)
        def response(model, body):
            prompt = body['messages'][-1]['content']
            text = ('候选3更合适。' if '只返回原始ID' in prompt else
                    '{"translation":"Moon."}' if body.get('response_format') else 'Moon.')
            return {'choices': [{'message': {'content': text}, 'finish_reason': 'stop'}]}
        with tempfile.TemporaryDirectory() as folder:
            config['generation_output'] = str(Path(folder) / 'translations.jsonl')
            config['realtime_steps_output'] = str(Path(folder) / 'steps.jsonl')
            path = Path(folder) / 'config.json'
            path.write_text(json.dumps(config), encoding='utf-8')
            args = ['generate.py', '--config', str(path), '--model', 'mimo', '--poems', 'T01']
            with contextlib.redirect_stdout(io.StringIO()), patch.object(generate, 'chat', side_effect=response) as caller, patch('sys.argv', args):
                generate.main()
                generate.main()
                self.assertEqual(caller.call_count, 6)
                self.assertEqual(caller.call_args.args[1]['response_format'], {'type': 'json_object'})
                self.assertEqual(caller.call_args.args[1]['temperature'], 0)
                self.assertEqual(caller.call_args.args[1]['messages'][-2],
                                 {'role': 'assistant', 'content': '候选3更合适。'})
            rows = list(generate.read_jsonl(config['generation_output']))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]['status'], 'ok')
            self.assertEqual(rows[0]['translation'], 'Moon.')
            self.assertEqual(rows[0]['generation_protocol']['version'], 'normal-flow-v4-selection-content-passthrough')
            self.assertIsNone(generate.protocol_issue(rows[0], {}))

    def test_normal_flow_applies_fixes_without_changing_resume_ids(self):
        config, root = generate.load_config('run_config.json')
        old = list(generate.jobs(config, root, 'mimo', ['G6-C07', 'G6-C09', 'G1-C11', 'G1-C01'], ['T01']))
        fixed = generate.normal_flow_work(old)
        self.assertEqual([r['id'] for r, _ in fixed], [r['id'] for r, _ in old])
        for (row, recipe), (old_row, original) in zip(fixed, old):
            if row['condition_id'] == 'G1-C01':
                self.assertEqual(recipe['steps'][0]['sample_slot'], original['steps'][0]['sample_slot'])
                self.assertEqual([s['prompt'] for s in recipe['steps']], [s['prompt'] for s in original['steps']])
                continue
            self.assertIn('generation_protocol', row)
            if row['condition_id'] == 'G6-C07':
                self.assertNotIn('select_from', recipe['steps'][-2])
                self.assertNotIn('ranking', recipe['steps'][-2])
                self.assertEqual(recipe['steps'][-1]['history_from'], recipe['steps'][-2]['id'])
                self.assertEqual(recipe['steps'][-1]['translation_format'], 'json_only')
            elif row['condition_id'] == 'G6-C09':
                self.assertEqual(recipe['steps'][0]['max_tokens'], 4096)
            else:self.assertEqual(recipe['steps'][1]['native_json_key'], 'feedback')

    def test_atomic_replacement_retries_temporary_windows_file_lock(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'results.jsonl'
            path.write_text('{"old":true}\n', encoding='utf-8')
            replace = Path.replace
            calls = []
            def locked_once(source, target):
                calls.append(target)
                if len(calls) == 1:raise PermissionError('reader briefly holds file')
                return replace(source, target)
            with patch.object(Path, 'replace', locked_once), patch.object(generate.time, 'sleep'):
                generate.write_jsonl_atomic(path, [{'new': True}])
            self.assertEqual(list(generate.read_jsonl(path)), [{'new': True}])
            self.assertEqual(len(calls), 2)

    def test_retry_replaces_only_failure_and_reuses_candidates(self):
        config, root = generate.load_config('run_config.json')
        config.update(conditions=['G6-C07', 'G2-C01'], repeats=1, experiment_files=[])
        config['normal_flow_fixes'] = False  # 人为复现旧失败后测试定向修复。
        for field in ('poems', 'design'):config[field] = str(root / config[field])
        config.pop('materials', None)
        def response(model, body):
            prompt = body['messages'][-1]['content']
            if '不是复述展示顺序' in prompt:text = '4,3,2,1'
            elif '只返回原始ID' in prompt:text = '4321'
            elif body.get('response_format'):text = '{"translation":"Moon."}'
            else:text = 'Moon.'
            return {'choices': [{'message': {'content': text}, 'finish_reason': 'stop'}]}
        with tempfile.TemporaryDirectory() as folder:
            config['generation_output'] = str(Path(folder) / 'translations.jsonl')
            config['realtime_steps_output'] = str(Path(folder) / 'steps.jsonl')
            path = Path(folder) / 'config.json'
            path.write_text(json.dumps(config), encoding='utf-8')
            args = ['generate.py', '--config', str(path), '--model', 'qwen', '--poems', 'T01', '--workers', '2']
            with contextlib.redirect_stdout(io.StringIO()), patch.object(generate, 'chat', side_effect=response) as caller:
                with patch('sys.argv', args):generate.main()
                before = list(generate.read_jsonl(config['generation_output']))
                self.assertEqual(sum(r['status'] == 'faild' for r in before), 1)
                calls = caller.call_count
                with patch('sys.argv', args + ['--retry-failed']):generate.main()
                after = list(generate.read_jsonl(config['generation_output']))
                self.assertEqual(caller.call_count - calls, 2)  # 四份候选复用，仅排序和最终输出调用。
                self.assertEqual(len(after), len(before))
                self.assertEqual({r['id'] for r in after}, {r['id'] for r in before})
                self.assertTrue(all(r['status'] == 'ok' for r in after))
                original = next(r for r in before if r['status'] == 'ok')
                self.assertEqual(next(r for r in after if r['id'] == original['id']), original)
                with patch('sys.argv', args + ['--retry-failed']):generate.main()
                self.assertEqual(caller.call_count - calls, 2)  # 再次执行不重抽成功项。

    def test_ranking_selects_original_id_not_display_position(self):
        step = {'select_from': ['a', 'b', 'c', 'd'], 'ranking': True}
        self.assertEqual(generate.select_result(step, '4,2,1,3', dict(a='A', b='B', c='C', d='D')), 'D')
        self.assertEqual(generate.select_result(step, '候选4，候选2，候选1，候选3', dict(a='A', b='B', c='C', d='D')), 'D')
        with self.assertRaises(ValueError):
            generate.select_result(step, '4,4,1,3', dict(a='A', b='B', c='C', d='D'))

    def test_feedback_json_keeps_thinking_disabled_and_extracts_text(self):
        step = {'id': 'feedback', 'prompt': '检查译文', 'native_json_key': 'feedback', 'result_json_key': 'feedback'}
        body = generate.stage_body(step, {}, {'provider': 'mimo', 'model': 'mimo-v2.6-flash', 'thinking_enabled': False}, 1)
        self.assertEqual(body['thinking'], {'type': 'disabled'})
        self.assertEqual(body['response_format'], {'type': 'json_object'})
        self.assertIn('feedback', body['messages'][0]['content'])
        self.assertEqual(generate.decode_step(step, '{"feedback":"无需修改"}', {})[0], '无需修改')

    def test_repair_only_changes_target_steps(self):
        config, root = generate.load_config('run_config.json')
        jobs = list(generate.jobs(config, root, 'qwen', ['G6-C07', 'G6-C09', 'G1-C11', 'G1-C01'], ['T01']))
        for row, recipe in jobs:
            fixed = generate.repair_recipe(row['condition_id'], recipe)
            if row['condition_id'] == 'G6-C07':
                self.assertEqual(fixed, recipe)  # 反序配置本身正确，不再额外要求完整排名。
            elif row['condition_id'] == 'G6-C09':
                self.assertEqual(fixed['steps'][0]['max_tokens'], 4096)
                self.assertEqual(fixed['steps'][1]['max_tokens'], 4096)
            elif row['condition_id'] == 'G1-C11':
                self.assertEqual(fixed['steps'][0], recipe['steps'][0])
                self.assertEqual(fixed['steps'][1]['native_json_key'], 'feedback')

