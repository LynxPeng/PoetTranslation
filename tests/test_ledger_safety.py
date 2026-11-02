import tempfile
import unittest
from pathlib import Path

import generate


class LedgerSafetyTests(unittest.TestCase):
    def test_invalid_jsonl_identifies_file_and_physical_line(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'steps.jsonl'
            path.write_text('{"id":"ok"}\nbroken fragment\n', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, r'steps\.jsonl:2'):
                list(generate.read_jsonl(path))

    def test_generation_lock_rejects_second_writer_and_releases(self):
        with tempfile.TemporaryDirectory() as folder:
            ledger = Path(folder) / 'steps.jsonl'
            with generate.generation_lock(ledger):
                with self.assertRaisesRegex(ValueError, '生成进程'):
                    with generate.generation_lock(ledger):
                        self.fail('second writer entered')
            with generate.generation_lock(ledger):
                pass


if __name__ == '__main__':
    unittest.main()
