import unittest

import inspect_translations


class TranslationInspectionTests(unittest.TestCase):
    def test_ascii_start_whitespace_and_exact_note_variants(self):
        rows = [
            {'generator_track': 'mimo', 'translation': 'Moonlight'},
            {'generator_track': 'mimo', 'translation': '  Moon\nNote: comment'},
            {'generator_track': 'qwen', 'translation': '“Moon”\nnote: a\nNOTE: b'},
            {'generator_track': 'qwen', 'translation': 'Évening\nnOtE: ignored'},
            {'generator_track': 'qwen', 'translation': '月光'},
        ]
        report = inspect_translations.inspect_rows(enumerate(rows, 1))
        self.assertEqual(report['summary']['non_english_start'], 3)
        self.assertEqual(report['summary']['raw_non_english_start'], 4)
        self.assertEqual(report['summary']['contains_note'], 2)
        self.assertEqual(report['summary']['note_occurrences'], 3)
        self.assertEqual(report['summary']['note_variants'], {'note:': 1, 'Note:': 1, 'NOTE:': 1})
        self.assertEqual(report['by_model']['qwen']['non_english_start'], 3)
        self.assertEqual(len(report['matches']), 4)

    def test_missing_and_empty_are_reported_separately(self):
        report = inspect_translations.inspect_rows(enumerate([{}, {'translation': ''}, {'translation': '  '}, {'translation': 3}], 1))
        self.assertEqual(report['summary']['records'], 4)
        self.assertEqual(report['summary']['invalid_translation'], 4)
        self.assertEqual(report['summary']['non_english_start'], 0)

