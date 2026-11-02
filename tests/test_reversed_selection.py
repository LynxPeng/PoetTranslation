"""中间选择原样传递content与候选上下文，只校验最终译文。"""
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import generate


class ReversedSelectionTests(unittest.TestCase):
    def setUp(self):
        path = Path(__file__).resolve().parents[1] / 'configs/groups/g6.json'
        self.raw = json.loads(path.read_text(encoding='utf-8'))['experiments']['G6-C07']
        self.recipe = generate.final_translation_recipe(
            generate.repair_recipe('G6-C07', self.raw), 'G6-C07')

    def test_reversed_display_passes_raw_content_then_formats_translation(self):
        selection, final = self.recipe['steps'][-2:]
        self.assertNotIn('ranking', selection)
        self.assertNotIn('select_from', selection)
        self.assertNotIn('selection_format', selection)
        prompt = selection['prompt']
        self.assertLess(prompt.index('候选4：'), prompt.index('候选1：'))
        values = {f'candidate{i}': f'Translation {i}\nSecond line' for i in range(1, 5)}
        for content in ('候选1', '候选3', '1', '候选3最合适，理由如下。', '3,1,4,2'):
            self.assertEqual(generate.select_result(selection, content, values), content)
        self.assertEqual(final['translation_format'], 'json_only')
        self.assertEqual(final['history_from'], selection['id'])
        self.assertNotIn('native_json_key', selection)

    def test_order_control_matches_forward_selector_except_display_order(self):
        path = Path(__file__).resolve().parents[1] / 'configs/groups/g6.json'
        forward = json.loads(path.read_text(encoding='utf-8'))['experiments']['G6-C04']
        forward = generate.final_translation_recipe(forward, 'G6-C04')
        left = forward['steps'][-2]
        right = self.recipe['steps'][-2]
        self.assertNotIn('select_from', left)
        self.assertNotIn('select_from', right)
        self.assertEqual(forward['steps'][:4], self.recipe['steps'][:4])

    def test_tree_branches_receive_selection_conversation(self):
        path = Path(__file__).resolve().parents[1] / 'configs/groups/g6.json'
        raw = json.loads(path.read_text(encoding='utf-8'))['experiments']['G6-C09']
        recipe = generate.final_translation_recipe(generate.repair_recipe('G6-C09', raw), 'G6-C09')
        steps = {s['id']: s for s in recipe['steps']}
        self.assertNotIn('select_from', steps['selected_plan'])
        for sid in ('tree_candidate1', 'tree_candidate2'):
            self.assertEqual(steps[sid]['history_from'], 'selected_plan')

    def test_real_flow_forwards_unparsed_selection_and_all_candidates(self):
        def response(content):
            return {'choices': [{'message': {'content': content}, 'finish_reason': 'stop'}]}
        model = {'provider': 'qwen', 'model': 'qwen3.8-flash', 'base_url': 'https://example.invalid/v1'}
        candidates = [f'Full candidate {i}\nLine two' for i in range(1, 5)]
        content = '我选择候选3，因为它保留了原作的意象。'
        replies = candidates + [content, json.dumps({'translation': candidates[2]})]
        with patch.object(generate, 'chat', side_effect=[response(s) for s in replies]) as caller:
            text, _ = generate.run_recipe(self.recipe, {'source': json.dumps({'text': '月'})}, model, seed=1)
        self.assertEqual(text, candidates[2])
        final = caller.call_args.args[1]
        self.assertEqual(final['response_format'], {'type': 'json_object'})
        self.assertEqual(final['messages'][-2], {'role': 'assistant', 'content': content})
        for candidate in candidates:self.assertIn(candidate, final['messages'][-3]['content'])
        self.assertNotIn('response_format', caller.call_args_list[-2].args[1])


if __name__ == '__main__':
    unittest.main()
