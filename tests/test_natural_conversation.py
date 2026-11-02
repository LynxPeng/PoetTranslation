import json
from pathlib import Path
import unittest
from unittest.mock import patch

import generate


class NaturalConversationTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(__file__).resolve().parents[1]
        self.model = {'provider': 'mimo', 'model': 'mimo-v2.6-flash', 'base_url': 'http://local/v1', 'thinking_enabled': True}

    @staticmethod
    def raw(content):
        return {'choices': [{'message': {'content': content, 'reasoning_content': 'private reasoning'}, 'finish_reason': 'stop'}]}

    def test_chain_carries_visible_dialogue_and_only_final_has_preamble(self):
        recipe = {'conversation_mode': True, 'steps': [
            {'id': 'draft', 'prompt': 'Translate {{source}}', 'output_format': 'translation', 'translation_format': 'plain'},
            {'id': 'feedback', 'prompt': 'Check the translation', 'history_from': 'draft', 'output_format': 'feedback_text'},
            {'id': 'final', 'prompt': 'Revise it', 'history_from': 'feedback', 'output_format': 'translation', 'translation_format': 'tag', 'output_preamble': generate.TRANSLATION_PREAMBLE}]}
        with patch.object(generate, 'chat', side_effect=[self.raw('Moon'), self.raw('One issue'), self.raw('<translation>Moonlight</translation>')]) as caller:
            translation, _ = generate.run_recipe(recipe, {'source': '{}'}, self.model, seed=1)
        self.assertEqual(translation, 'Moonlight')
        bodies = [call.args[1] for call in caller.call_args_list]
        self.assertEqual([len(body['messages']) for body in bodies], [1, 3, 5])
        self.assertEqual(bodies[1]['messages'][1], {'role': 'assistant', 'content': 'Moon'})
        self.assertEqual(bodies[2]['messages'][3], {'role': 'assistant', 'content': 'One issue'})
        self.assertNotIn('private reasoning', json.dumps(bodies))
        self.assertTrue(bodies[-1]['messages'][-1]['content'].startswith(generate.TRANSLATION_PREAMBLE))
        self.assertNotIn('<要求>', bodies[0]['messages'][0]['content'])

    def test_active_config_prompts_have_no_json_notes_or_stage_jargon(self):
        config, root = generate.load_config(self.root / 'run_config.json')
        work = list(generate.jobs(config, root, 'mimo', poem_ids=['T01']))
        for _, recipe in work:
            calls = [s for s in recipe['steps'] if s.get('op', 'llm') == 'llm']
            self.assertTrue(recipe.get('conversation_mode'))
            for index, step in enumerate(calls):
                for forbidden in ('JSON', 'notes', 'Estimate阶段', 'TEaR'):
                    self.assertNotIn(forbidden, step['prompt'])
                if index != len(calls) - 1:self.assertFalse(step.get('output_preamble'))
                if step.get('translation_format') == 'plain':self.assertIn('只给英文译文正文', step['prompt'])
        baseline = next(rec for row, rec in work if row['condition_id'] == 'G1-C00')
        self.assertEqual(baseline['steps'][0]['translation_format'], 'json_only')
        self.assertNotIn('output_preamble',baseline['steps'][0])

    def test_plain_feedback_stop_marker_is_unambiguous(self):
        self.assertFalse(generate.gate_passes({'issues_present': 'feedback'}, {'feedback': '无需修改。'}))
        self.assertTrue(generate.gate_passes({'issues_present': 'feedback'}, {'feedback': '有遗漏，需要修正。'}))
        self.assertTrue(generate.gate_passes({'issues_present': 'feedback'}, {'feedback': '部分地方无需修改，但有一处误译。'}))

    def recipe(self, condition):
        config, root = generate.load_config(self.root / 'run_config.json')
        config['repeats'] = 1
        config['final_json_output'] = False
        row, recipe = next(generate.jobs(config, root, 'mimo', [condition], ['T01']))
        values = dict(row['_bindings'], source=json.dumps(row['source'], ensure_ascii=False))
        return row, recipe, values

    def test_adaptive_stop_uses_existing_translation_without_extra_call(self):
        row, recipe, values = self.recipe('G1-C12')
        for responses, expected in ((['Moon', '无需修改。'], 'Moon'),
                                    (['Moon', '有遗漏', 'Moonlight', '无需修改'], 'Moonlight')):
            with patch.object(generate, 'chat', side_effect=[self.raw(s) for s in responses]) as caller:
                translation, trace = generate.run_recipe(recipe, values, self.model, seed=1, context=row)
            self.assertEqual(translation, expected)
            self.assertEqual(caller.call_count, len(responses))
            self.assertEqual(trace[-1]['recipe_result']['formatted_result'], '<translation>' + expected + '</translation>')

    def test_backtranslation_is_blind_to_original_and_revision_has_history(self):
        row, recipe, values = self.recipe('G1-C08')
        with patch.object(generate, 'chat', side_effect=[self.raw(s) for s in ('English draft', '中文回译', '有遗漏', '<translation>Revised</translation>')]) as caller:
            text, _ = generate.run_recipe(recipe, values, self.model, seed=1, context=row)
        bodies = [c.args[1] for c in caller.call_args_list]
        self.assertEqual(text, 'Revised')
        self.assertEqual(len(bodies[1]['messages']), 1)
        self.assertIn('English draft', bodies[1]['messages'][0]['content'])
        self.assertNotIn(row['source']['text'], bodies[1]['messages'][0]['content'])
        self.assertEqual(len(bodies[-1]['messages']), 5)

    def test_candidate_branches_remain_independent_and_selection_is_local(self):
        row, recipe, values = self.recipe('G6-C02')
        with patch.object(generate, 'chat', side_effect=[self.raw(s) for s in ('Candidate A', 'Candidate B', '2')]) as caller:
            text, trace = generate.run_recipe(recipe, values, self.model, seed=1, context=row)
        bodies = [c.args[1] for c in caller.call_args_list]
        self.assertEqual([len(body['messages']) for body in bodies], [1, 1, 1])
        self.assertNotIn('Candidate A', bodies[1]['messages'][0]['content'])
        self.assertEqual(text, 'Candidate B')
        self.assertEqual(trace[-1]['recipe_result']['formatted_result'], '<translation>Candidate B</translation>')

    def test_cached_chain_rebuilds_dialogue_without_new_calls(self):
        row, recipe, values = self.recipe('G1-C06')
        cache = {}
        with patch.object(generate, 'chat', side_effect=[self.raw(s) for s in ('Moon', '有遗漏', '<translation>Moonlight</translation>')]) as caller:
            generate.run_recipe(recipe, values, self.model, seed=1, context=row, cache=cache)
            text, trace = generate.run_recipe(recipe, values, self.model, seed=1, context=row, cache=cache)
        self.assertEqual(text, 'Moonlight')
        self.assertEqual(caller.call_count, 3)
        self.assertTrue(all(t.get('reused') for t in trace if 'step_record_id' in t))


if __name__ == '__main__':unittest.main()
