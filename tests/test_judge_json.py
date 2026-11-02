import json
import unittest
from pathlib import Path

import evaluate
import generate


class JudgeJsonTests(unittest.TestCase):
    packet = {'D1': 25, 'D2': 8, 'D3': 12, 'D4': 13, 'D5': 9, 'D6': 8.5, 'D7': 8, 'total': 83.5}

    def test_scores_ranges_precision_and_total_are_checked(self):
        self.assertEqual(evaluate.parse_score(json.dumps(self.packet)), self.packet)
        for changes in ({'D1': 31}, {'D2': True}, {'D3': -1}, {'D4': '13'},
                        {'D5': float('nan')}, {'D6': 8.55}, {'total': 83.6}, {'extra': 0}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                evaluate.parse_score(json.dumps(dict(self.packet, **changes)))
        for invalid in ('83.5', '{}', json.dumps({k: v for k, v in self.packet.items() if k != 'D7'})):
            with self.assertRaises(ValueError):evaluate.parse_score(invalid)

    def test_all_poems_share_exact_system_prompt_translation_is_last(self):
        root = Path(__file__).resolve().parents[1]
        rubric = generate.read_json(root / 'evaluation/rubric.json')
        instructions = (root / 'evaluation/judge_prompt.md').read_text(encoding='utf-8').split('## A. 高优先级指令正文', 1)[1].split('## B.', 1)[0].strip()
        systems = set()
        for poem in generate.read_json(root / 'data/benchmark/poems.json')['poems']:
            for translation in ('first anonymous translation', 'second anonymous translation'):
                messages = evaluate.judge_messages(poem, rubric, translation, instructions)
                systems.add(messages[0]['content'])
                self.assertEqual(list(json.loads(messages[1]['content'])), ['source', 'translation'])
                self.assertNotIn('generator_track', json.dumps(messages))
        self.assertEqual(len(systems), 1)
        self.assertIn('JSON', next(iter(systems)))

    def test_sort_groups_poems_and_lexically_similar_translations(self):
        def entry(system, poem, key, translation):
            return ({'poem_id': poem, 'id': key, 'translation': translation}, {'messages': [{'role': 'system', 'content': system}]}, key, key)
        rows = [entry('z', 'S01', '1', 'A'), entry('a', 'T01', '2', 'A'),
                entry('a', 'S01', '3', 'Moon bright'), entry('a', 'S01', '4', 'Moon above'),
                entry('a', 'S01', '5', 'Before my bed')]
        self.assertEqual([x[0]['id'] for x in sorted(rows, key=evaluate.evaluation_order)], ['5', '4', '3', '2', '1'])

