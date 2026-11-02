import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import generate


class RealtimeGenerationTests(unittest.TestCase):
    def setUp(self):
        self.config, self.root = generate.load_config(Path(__file__).resolve().parents[1] / 'run_config.json')
        self.config.update(transport='realtime', single_turn_source_first=True, repeats=2, final_json_output=False)
        self.model = generate.model_config(self.config, 'mimo')

    @staticmethod
    def response(text):
        return {'choices': [{'message': {'content': text}, 'finish_reason': 'stop'}]}

    def test_single_poem_prefix_and_multiturn_prompt_preservation(self):
        selected = ['G1-C00', 'G4-C01', 'G6-C12', 'G1-C06']
        for row, recipe in generate.jobs(self.config, self.root, 'mimo', selected, ['T01']):
            original = generate.resolve_recipe(self.config, row['condition_id'])
            if row['condition_id'] == 'G1-C06':
                self.assertEqual(recipe, generate.conversation_recipe(original))
                continue
            values = dict(row['_bindings'], source=json.dumps(row['source'], ensure_ascii=False))
            content = generate.stage_body(recipe['steps'][0], values, self.model, row['run_id'])['messages'][0]['content']
            self.assertTrue(content.startswith(generate.TRANSLATION_PREAMBLE + '<原诗>\n' + values['source'] + '\n</原诗>\n\n<任务>\n'))
            self.assertEqual(content.count(values['source']), 1)
            self.assertNotIn('{{', content)
        self.assertFalse(generate.resolve_recipe(self.config, 'G1-C00')['steps'][0]['prompt'].startswith('<原诗>'))

    def test_single_turn_poem_major_and_multiturn_condition_major(self):
        work = list(generate.jobs(self.config, self.root, 'mimo',
                                 ['G1-C06', 'G1-C00', 'G5-C01'], ['T01', 'T02']))
        scheduled = generate.schedule_jobs(work)
        actual = [(r['condition_id'], r['poem_id'], r['run_id']) for r, _ in scheduled]
        expected = [(c, p, n) for p in ('T01', 'T02') for c in ('G1-C00', 'G5-C01') for n in (1, 2)]
        expected += [('G1-C06', p, n) for p in ('T01', 'T02') for n in (1, 2)]
        self.assertEqual(actual, expected)
        self.assertEqual({r['id'] for r, _ in work}, {r['id'] for r, _ in scheduled})

    def test_multiturn_does_not_run_later_independent_step_early(self):
        recipe = {'steps': [{'id': 'a', 'prompt': 'first'},
                            {'id': 'b', 'prompt': 'second {{a}}'},
                            {'id': 'c', 'prompt': 'third independent'}]}
        with patch.object(generate, 'chat', return_value=self.response('text')) as caller:
            generate.run_recipe(recipe, {'source': '{}'}, self.model)
        prompts = [c.args[1]['messages'][0]['content'] for c in caller.call_args_list]
        self.assertEqual(prompts, ['first', 'second text', 'third independent'])

    def test_shared_steps_persist_but_runs_remain_independent(self):
        recipe = {'steps': [{'id': 'draft', 'sample_slot': 'initial', 'prompt': '{{source}}'}]}
        with tempfile.TemporaryDirectory() as folder:
            ledger = Path(folder) / 'steps.jsonl'
            cache = {}
            with patch.object(generate, 'chat', return_value=self.response('Moon')) as caller:
                for run in (1, 1, 2):
                    generate.run_recipe(recipe, {'source': '{}'}, self.model, seed=run,
                                        cache=cache, ledger=ledger)
                self.assertEqual(caller.call_count, 2)
            persisted = {r['id']: r for r in generate.read_jsonl(ledger)}
            with patch.object(generate, 'chat') as caller:
                _, trace = generate.run_recipe(recipe, {'source': '{}'}, self.model, seed=1,
                                               cache=persisted, ledger=ledger)
                caller.assert_not_called()
            self.assertTrue(trace[0]['reused'])
            self.assertEqual(len(persisted), 2)

    def test_interrupted_call_blocks_automatic_resubmission(self):
        recipe = {'steps': [{'id': 'x', 'prompt': '{{source}}'}]}
        row = {'id': 'a', 'poem_id': 'inline', 'run_id': 1, 'source': {}}
        ready, _ = generate.batch_wave([(row, recipe)], self.model, {})
        cache = {k: dict(v, status='running') for k, v in ready.items()}
        with patch.object(generate, 'chat') as caller, self.assertRaisesRegex(ValueError, '未确认'):
            generate.run_recipe(recipe, {'source': '{}'}, self.model, seed=1, cache=cache)
        caller.assert_not_called()

    def test_realtime_cli_order_and_resume_with_two_poems(self):
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
            config = dict(self.config)
            config.update(translation_tags=False, natural_conversations=False, generation_workers=1)
            for key in ('result_json_templates', 'output_templates', 'batch_generation_output'):config.pop(key, None)
            config['conditions'] = ['G1-C00', 'G5-C01', 'G1-C06']
            config.pop('materials')
            config['poems'] = str(self.root / config['poems'])
            config['design'] = str(self.root / config['design'])
            config['generation_output'] = str(Path(folder) / 'translations.jsonl')
            config['realtime_steps_output'] = str(Path(folder) / 'steps.jsonl')
            config['experiments'] = {
                'G1-C00': {'steps': [{'id': 'a', 'prompt': 'Translate A {{source}}'}]},
                'G5-C01': {'steps': [{'id': 'b', 'prompt': 'Translate B {{source}}'}]},
                'G1-C06': {'steps': [{'id': 'draft', 'prompt': 'Draft {{source}}'},
                                      {'id': 'final', 'prompt': 'Revise {{draft}} {{source}}'}]}}
            path = Path(folder) / 'config.json'
            path.write_text(json.dumps(config, ensure_ascii=False), encoding='utf-8')
            # 第一首的首个条件已完成；续跑仍应先补齐第一首的其余单轮条件。
            work = list(generate.jobs(config, path.parent, 'mimo', poem_ids=['T01', 'T02']))
            for row, _ in work[:2]:
                generate.append_jsonl(config['generation_output'],
                                      dict({k: v for k, v in row.items() if not k.startswith('_')},
                                           status='ok', translation='Earlier'))
            with patch('sys.argv', ['generate.py', '--config', str(path), '--poems', 'T01', 'T02']), \
                    patch.object(generate, 'chat', return_value=self.response('Moon')) as caller:
                generate.main()
                count = caller.call_count
                generate.main()
                self.assertEqual(caller.call_count, count)
            rows = list(generate.read_jsonl(config['generation_output']))
            expected = [(c, p, n) for p in ('T01', 'T02') for c in ('G1-C00', 'G5-C01') for n in (1, 2)]
            expected += [('G1-C06', p, n) for p in ('T01', 'T02') for n in (1, 2)]
            self.assertEqual([(r['condition_id'], r['poem_id'], r['run_id']) for r in rows],
                             expected)
            self.assertEqual(len(rows), 12)
            self.assertTrue(all(r['status'] == 'ok' for r in rows))

    def test_unconfirmed_cli_call_is_not_resent_and_other_jobs_continue(self):
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
            config = dict(self.config)
            config.update(translation_tags=False, natural_conversations=False, generation_workers=1)
            for key in ('result_json_templates', 'output_templates', 'batch_generation_output'):config.pop(key, None)
            config.pop('materials')
            config['poems'] = str(self.root / config['poems'])
            config['design'] = str(self.root / config['design'])
            config['conditions'] = ['G1-C00']
            config['experiments'] = {'G1-C00': {'steps': [{'id': 'x', 'prompt': '{{source}}'}]}}
            output, ledger = Path(folder) / 'translations.jsonl', Path(folder) / 'steps.jsonl'
            config.update(generation_output=str(output), realtime_steps_output=str(ledger))
            path = Path(folder) / 'config.json'
            path.write_text(json.dumps(config), encoding='utf-8')
            row, recipe = next(generate.jobs(config, path.parent, 'mimo', poem_ids=['T01']))
            ready, _ = generate.batch_wave([(row, recipe)], self.model, {})
            entry = next(iter(ready.values()))
            generate.append_jsonl(ledger, dict(entry, status='running'))
            with patch('sys.argv', ['generate.py', '--config', str(path), '--poems', 'T01']), \
                    patch.object(generate, 'chat', return_value=self.response('Moon')) as caller:
                for _ in range(2):
                    generate.main()
                self.assertEqual(caller.call_count, 1)
            rows = list(generate.read_jsonl(output))
            self.assertEqual([r['status'] for r in rows], ['faild', 'ok'])
            self.assertIsNone(rows[0]['content'])
            self.assertEqual(rows[0]['error_kind'], 'UnconfirmedCallError')
            self.assertEqual(rows[1]['translation'], 'Moon')


if __name__ == '__main__':
    unittest.main()
