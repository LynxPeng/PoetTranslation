import importlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


class CombinationTests(unittest.TestCase):
    def module(self):
        try:
            return importlib.import_module('run_combination')
        except ModuleNotFoundError:
            self.fail('推荐组合执行模块尚未实现')

    def test_only_pretranslation_analysis_has_search_tool(self):
        m = self.module()
        model = {'provider': 'mimo', 'model': 'mimo-v2.6-flash', 'thinking_enabled': True}
        body = m.make_body(model, [{'role': 'user', 'content': '原诗'}], 8192, phase='analysis')
        self.assertEqual(body['thinking'], {'type': 'enabled'})
        self.assertEqual(body['tools'][0]['type'], 'function')
        self.assertEqual(body['tools'][0]['function']['name'], 'web_search')
        self.assertNotIn('temperature', body)
        self.assertNotIn('response_format', body)
        for phase in ('generation', 'select', 'estimate'):
            self.assertNotIn('tools', m.make_body(model, [], 8192, phase=phase))

    def test_only_final_is_translation_json(self):
        m = self.module()
        model = {'provider': 'mimo', 'model': 'mimo', 'thinking_enabled': True}
        messages = [{'role': 'user', 'content': '修订'}]
        body = m.make_body(model, messages, 8192, final=True)
        self.assertEqual(body['response_format']['type'], 'json_schema')
        self.assertNotIn('tools', body)
        self.assertEqual(body['thinking']['type'], 'enabled')
        self.assertEqual(len(messages), 1)

    def test_qwen_candidates_have_no_search_tools(self):
        m = self.module()
        body = m.make_body({'provider': 'qwen', 'model': 'qwen3.8-flash',
                            'thinking_enabled': True, 'thinking_budget': 4096}, [], 12000)
        self.assertNotIn('tools', body)
        self.assertEqual(body['thinking_budget'], 4096)
        self.assertEqual(body['max_tokens'], 12000)
        self.assertIn('messages', body)
        self.assertNotIn('search_strategy', body)

    def test_responses_parser_ignores_tool_and_reasoning_text(self):
        m = self.module()
        raw = {'status': 'completed', 'output': [
            {'type': 'reasoning', 'content': [{'type': 'reasoning_text', 'text': '秘密思考'}]},
            {'type': 'web_search_call', 'action': {'query': '典故'}},
            {'type': 'message', 'role': 'assistant', 'content': [
                {'type': 'output_text', 'text': 'Moonlight\nHome'}]}],
            'usage': {'input_tokens': 100, 'output_tokens': 20,
                      'input_tokens_details': {'cached_tokens': 80}}}
        normalized = m.normalize_response(raw)
        self.assertEqual(m.response_text(normalized, allow_thinking=True), 'Moonlight\nHome')
        self.assertEqual(normalized['usage']['prompt_tokens_details']['cached_tokens'], 80)
        self.assertEqual(normalized['_provider_response'], raw)

    def test_incomplete_responses_are_not_accepted(self):
        m = self.module()
        with self.assertRaises(ValueError):
            m.normalize_response({'status': 'incomplete', 'output': []})

    def test_responses_reasoning_nullable_content(self):
        m = self.module()
        raw = {'status': 'completed', 'output': [
            {'type': 'reasoning', 'content': None, 'summary': []},
            {'type': 'message', 'content': [{'type': 'output_text', 'text': 'Moon'}], 'summary': None}]}
        normalized = m.normalize_response(raw)
        self.assertEqual(normalized['choices'][0]['message']['content'], 'Moon')

    def test_examples_have_only_source_and_translation(self):
        m = self.module()
        root = Path(__file__).resolve().parents[1]
        examples = m.load_examples(root / 'data/experiment/examples.json')
        self.assertEqual(len(examples), 4)
        self.assertTrue(all(set(e) == {'source', 'translation'} for e in examples))
        self.assertNotIn('error_ledger', json.dumps(examples))

    def test_pipeline_keeps_all_candidates_and_does_not_parse_selection(self):
        m = self.module()
        config = m.read_json(Path(__file__).resolve().parents[1] / 'run_combination.json')
        poem = {'id': 'T01', 'title': '诗', 'author': '人', 'text': '原文'}
        seen = []
        def fake(step, track, messages, final=False):
            seen.append((step, track, messages, final))
            content = '{"translation":"Moonlight\\nHome"}' if final else step + '正文，不解析任何排序编号'
            return {'role': 'assistant', 'content': content, 'reasoning_content': '保留思考'}
        result = m.translate_poem(poem, config, [], fake)
        self.assertEqual(result, 'Moonlight\nHome')
        self.assertEqual(len(seen), 8)
        candidates = [x for x in seen if x[0].startswith('candidate_')]
        self.assertEqual(len(candidates), 4)
        self.assertEqual(sum(x[1] == 'mimo' for x in candidates), 2)
        self.assertEqual(sum(x[1] == 'qwen38' for x in candidates), 2)
        for step, _, messages, _ in candidates:
            self.assertFalse(any('candidate_' in x['content'] for x in messages))
        selection = next(x for x in seen if x[0] == 'select')
        joined = json.dumps(selection[2], ensure_ascii=False)
        for candidate in candidates:
            self.assertIn(candidate[0] + '正文', joined)
        self.assertIn('综合', joined)
        self.assertIn('不必照搬', joined)
        last = seen[-1]
        self.assertTrue(last[3])
        self.assertIn('select正文', json.dumps(last[2], ensure_ascii=False))
        self.assertIn('estimate正文', json.dumps(last[2], ensure_ascii=False))
        self.assertEqual(sum(x[3] for x in seen), 1)

    def test_completed_request_reused_and_sample_slots_distinct(self):
        m = self.module()
        with tempfile.TemporaryDirectory() as folder:
            cfg = {'models': {'mimo': {'provider': 'mimo', 'model': 'mimo',
                                      'base_url': 'https://example.test', 'thinking_enabled': True}},
                   'max_output_tokens': 10000, 'version': 'test'}
            path = Path(folder) / 'steps.jsonl'
            raw = {'choices': [{'finish_reason': 'stop', 'message': {'content': '正文'}}]}
            with patch.object(m, 'provider_call', return_value=raw) as api:
                ledger = m.StepLedger(cfg, path)
                messages = [{'role': 'user', 'content': '相同提示'}]
                ledger.call('T01', 'candidate_1', 'mimo', messages)
                ledger.call('T01', 'candidate_1', 'mimo', messages)
                ledger.call('T01', 'candidate_2', 'mimo', messages)
                self.assertEqual(api.call_count, 2)
                resumed = m.StepLedger(cfg, path)
                resumed.call('T01', 'candidate_1', 'mimo', messages)
                self.assertEqual(api.call_count, 2)

    def test_failed_and_unknown_calls_not_silently_retried(self):
        m = self.module()
        with tempfile.TemporaryDirectory() as folder:
            cfg = {'models': {'mimo': {'provider': 'mimo', 'model': 'mimo',
                                      'base_url': 'https://example.test', 'thinking_enabled': True}},
                   'max_output_tokens': 10000, 'version': 'test'}
            path = Path(folder) / 'steps.jsonl'
            with patch.object(m, 'provider_call', side_effect=RuntimeError('fail')) as api:
                ledger = m.StepLedger(cfg, path)
                for _ in range(2):
                    with self.assertRaises(RuntimeError):
                        ledger.call('T01', 'analysis', 'mimo', [])
                self.assertEqual(api.call_count, 1)
            rows = list(m.read_jsonl(path))
            m.append_jsonl(path, dict(rows[0], status='running'))
            with patch.object(m, 'provider_call') as api:
                with self.assertRaises(m.UnconfirmedCallError):
                    m.StepLedger(cfg, path).call('T01', 'analysis', 'mimo', [])
                api.assert_not_called()

    def test_unexecuted_search_is_resolved_before_next_stage(self):
        m = self.module()
        with tempfile.TemporaryDirectory() as folder:
            cfg = {'models': {}, 'max_output_tokens': 10000, 'version': 'test'}
            ledger = m.StepLedger(cfg, Path(folder) / 'steps.jsonl')
            answers = [
                {'role': 'assistant', 'content': '<tool_call><function=web_search><parameter=query>床 词义</parameter></function></tool_call>',
                 'reasoning_content': '思考'},
                {'role': 'assistant', 'content': '词典来源：床可以指卧具。', 'annotations': [{'url': 'https://example.org'}]},
                {'role': 'assistant', 'content': '核实后的简短分析'}]
            with patch.object(ledger, '_request', side_effect=answers) as request:
                value = ledger.call('T01', 'analysis', 'mimo', [{'role': 'user', 'content': '原诗'}])
            self.assertEqual(value['content'], '核实后的简短分析')
            self.assertEqual(request.call_count, 3)
            followup = request.call_args_list[-1].args[3]
            self.assertIn('词典来源', json.dumps(followup, ensure_ascii=False))
            self.assertIn('reasoning_content', followup[1])

    def test_structured_search_call_arguments_are_extracted(self):
        m = self.module()
        queries = m.pending_searches({'tool_calls': [{'id': 'abc', 'function': {
            'name': 'web_search', 'arguments': '{"query":"返景 词义"}'}}]})
        self.assertEqual(queries, [('abc', '返景 词义')])

    def test_bad_final_is_failed_in_ledger_and_can_retry_only_that_request(self):
        m = self.module()
        with tempfile.TemporaryDirectory() as folder:
            cfg = {'models': {'mimo': {'provider': 'mimo', 'model': 'mimo',
                                      'base_url': 'https://example.test', 'thinking_enabled': True}},
                   'max_output_tokens': 10000, 'version': 'test'}
            path = Path(folder) / 'steps.jsonl'
            def raw(content):
                return {'choices': [{'finish_reason': 'stop', 'message': {'content': content}}]}
            with patch.object(m, 'provider_call', side_effect=[raw('not json'), raw('{"translation":"Moon"}')]) as api:
                with self.assertRaises(ValueError):
                    m.StepLedger(cfg, path).call('T01', 'final', 'mimo', [], final=True)
                self.assertEqual(list(m.read_jsonl(path))[-1]['status'], 'error')
                result = m.StepLedger(cfg, path, retry_failed=True).call('T01', 'final', 'mimo', [], final=True)
                self.assertEqual(json.loads(result['content'])['translation'], 'Moon')
                self.assertEqual(api.call_count, 2)

    def test_malformed_tool_request_is_failed_not_cached_success(self):
        m = self.module()
        with tempfile.TemporaryDirectory() as folder:
            cfg = {'models': {'mimo': {'provider': 'mimo', 'model': 'mimo',
                                      'base_url': 'https://example.test', 'thinking_enabled': True}},
                   'max_output_tokens': 10000, 'version': 'test'}
            path = Path(folder) / 'steps.jsonl'
            raw = {'choices': [{'finish_reason': 'stop', 'message': {'content': '<tool_call>broken</tool_call>'}}]}
            with patch.object(m, 'provider_call', return_value=raw):
                with self.assertRaises(ValueError):
                    m.StepLedger(cfg, path).call('T01', 'analysis', 'mimo', [])
            self.assertEqual(list(m.read_jsonl(path))[-1]['status'], 'error')

    def test_repair_with_updated_request_keeps_one_final_row_per_poem(self):
        m = self.module()
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'poems.json').write_text(json.dumps({'poems': [
                {'id': 'T01', 'title': '诗', 'author': '人', 'text': '原文'}]}), encoding='utf-8')
            cfg = {'version': 'test', 'condition_id': 'COMBO-C01', 'models': {},
                   'poems': 'poems.json', 'examples': 'unused.json', 'generation_output': 'translations.jsonl',
                   'realtime_steps_output': 'steps.jsonl', 'repeats': 1, 'candidates_per_model': 2}
            config_path = root / 'config.json'
            config_path.write_text(json.dumps(cfg), encoding='utf-8')
            m.append_jsonl(root / 'translations.jsonl', {'id': 'old', 'poem_id': 'T01', 'status': 'faild'})
            with patch.object(m.sys, 'argv', ['run_combination.py', '--config', str(config_path), '--retry-failed']), \
                 patch.object(m, 'load_examples', return_value=[]), \
                 patch.object(m, 'translate_poem', return_value='Moon'):
                m.main()
            rows = list(m.read_jsonl(root / 'translations.jsonl'))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]['status'], 'ok')
            self.assertNotEqual(rows[0]['id'], 'old')
            self.assertIn('timestamp', rows[0])

    def test_global_search_budget_is_persistent_and_thread_safe(self):
        m = self.module()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'search_budget.jsonl'
            budget = m.SearchBudget(path, maximum=16)
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=8) as pool:
                reserved = list(pool.map(lambda i: budget.reserve(f'T{i:02}', f'query{i}'), range(32)))
            self.assertEqual(sum(reserved), 16)
            self.assertFalse(m.SearchBudget(path, maximum=16).reserve('T99', 'another'))

    def test_search_is_one_shot_short_and_only_one_query_per_poem(self):
        m = self.module()
        with tempfile.TemporaryDirectory() as folder:
            budget = m.SearchBudget(Path(folder) / 'budget.jsonl', maximum=16)
            self.assertTrue(budget.reserve('T01', 'first'))
            self.assertFalse(budget.reserve('T01', 'second'))
            self.assertFalse(budget.reserve('T01', 'first'))
        body = m.make_body({'provider':'mimo','model':'mimo','thinking_enabled':True}, [], 16384, phase='search')
        self.assertEqual(body['tools'][0]['max_keyword'], 1)
        self.assertEqual(body['tools'][0]['limit'], 1)
        self.assertTrue(body['tools'][0]['force_search'])
        self.assertLessEqual(body['max_completion_tokens'], 2048)

    def test_completed_analysis_with_failed_search_resumes_without_new_calls(self):
        m = self.module()
        with tempfile.TemporaryDirectory() as folder:
            cfg = {'models': {}, 'max_output_tokens': 10000, 'version': 'test'}
            path = Path(folder) / 'steps.jsonl'
            answers = [
                {'role': 'assistant', 'content': '<tool_call><function=web_search><parameter=query>床</parameter></function></tool_call>'},
                RuntimeError('lookup failed'), {'role': 'assistant', 'content': '保留不确定的最终简短分析'}]
            ledger = m.StepLedger(cfg, path)
            with patch.object(ledger, '_request', side_effect=answers) as api:
                first = ledger.call('T01', 'analysis', 'mimo', [])
            self.assertEqual(api.call_count, 3)
            with patch.object(m.StepLedger, '_request') as api:
                second = m.StepLedger(cfg, path).call('T01', 'analysis', 'mimo', [])
            self.assertEqual(first, second)
            api.assert_not_called()

    def test_cached_previous_evidence_disables_new_analysis_search(self):
        m = self.module()
        with tempfile.TemporaryDirectory() as folder:
            cfg = {'models': {'mimo': {'provider': 'mimo', 'model': 'mimo',
                   'base_url': 'https://example.test', 'thinking_enabled': True}},
                   'max_output_tokens': 10000, 'version': 'test'}
            raw = {'choices': [{'finish_reason': 'stop', 'message': {'content': '复用分析'}}]}
            with patch.object(m, 'previous_evidence', return_value=[{'content': '已核实资料'}]), \
                 patch.object(m, 'provider_call', return_value=raw) as api:
                m.StepLedger(cfg, Path(folder) / 'steps.jsonl').call('T01', 'analysis', 'mimo', [])
            self.assertNotIn('tools', api.call_args.args[1])

    def test_all_poems_finish_research_before_any_candidates(self):
        m = self.module()
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            poems = [{'id': f'T0{i}', 'title': '诗', 'author': '人', 'text': '原文'} for i in (1, 2)]
            (root / 'poems.json').write_text(json.dumps({'poems': poems}), encoding='utf-8')
            cfg = {'version':'test','condition_id':'COMBO-C02','models':{},'poems':'poems.json',
                   'examples':'unused.json','generation_output':'translations.jsonl',
                   'realtime_steps_output':'steps.jsonl','repeats':1,'candidates_per_model':2}
            config_path = root / 'config.json'
            config_path.write_text(json.dumps(cfg), encoding='utf-8')
            events = []
            def translate(poem, *args, **kwargs):
                research = kwargs.get('research_only', False)
                events.append(('research' if research else 'candidate', poem['id']))
                return None if research else 'Moon'
            with patch.object(m.sys, 'argv', ['run_combination.py','--config',str(config_path)]), \
                 patch.object(m, 'load_examples', return_value=[]), patch.object(m, 'translate_poem', side_effect=translate):
                m.main()
            self.assertEqual(sum(e[0]=='research' for e in events), 2)
            self.assertLess(max(i for i,e in enumerate(events) if e[0]=='research'),
                            min(i for i,e in enumerate(events) if e[0]=='candidate'))

    def test_audit_accepts_real_native_search_sources_without_extra_search(self):
        from summarize_combination import check_tool_completion
        raw = {'choices':[{'message':{'content':'<tool_call><function=web_search><parameter=query>另一问题</parameter></function></tool_call>',
               'annotations':[{'url':'https://example.org','summary':'实际已取得的原文资料'}]}}],
               'usage':{'web_search_usage':{'tool_usage':1}}}
        check_tool_completion('analysis.search.abc', raw, [])

    def test_audit_rejects_unexecuted_search_without_real_sources(self):
        from summarize_combination import check_tool_completion
        raw = {'choices':[{'message':{'content':'<tool_call><function=web_search><parameter=query>问题</parameter></function></tool_call>'}}],
               'usage':{'web_search_usage':{'tool_usage':0}}}
        with self.assertRaises(AssertionError):
            check_tool_completion('analysis.search.abc', raw, [])


if __name__ == '__main__':
    unittest.main()
