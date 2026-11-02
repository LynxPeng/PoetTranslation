import unittest
import generate


class FinalJSONModeTests(unittest.TestCase):
    def test_only_final_translation_uses_json_mode(self):
        recipe = generate.conversation_recipe({'steps': [
            {'id':'draft','prompt':'Translate {{source}}','output_format':'translation'},
            {'id':'final','prompt':'Revise','history_from':'draft','output_format':'translation'}]}, final_json=True)
        for provider in ('mimo','qwen'):
            model={'provider':provider,'model':'mimo-v2.6-flash' if provider=='mimo' else 'qwen-flash','thinking_enabled':True}
            if provider=='qwen':model['thinking_budget']=1
            first=generate.stage_body(recipe['steps'][0],{'source':'{}'},model,1)
            last=generate.stage_body(recipe['steps'][1],{},model,1,first['messages']+[{'role':'assistant','content':'Moon'}])
            self.assertNotIn('response_format',first)
            self.assertEqual(last['response_format'],{'type':'json_object'})
            self.assertEqual(last['messages'][0],generate.TRANSLATION_JSON_MESSAGE)
            self.assertNotIn('<translation>',str(last))
            if provider=='qwen':
                self.assertFalse(last['enable_thinking'])
                self.assertNotIn('thinking_budget',last)
            else:self.assertEqual(last['thinking'],{'type':'enabled'})

    def test_only_translation_field_is_accepted(self):
        step={'output_format':'translation','translation_format':'json_only'}
        self.assertEqual(generate.decode_step(step,'{"translation":"Moon\\nHome"}',{}),('Moon\nHome',[]))
        self.assertEqual(generate.decode_step(step, '{"translation":"Moon","notes":[]}', {})[0], 'Moon')
        for text in ('{"translation":""}','{"translation":1}', '<translation>Moon</translation>'):
            with self.assertRaises(ValueError):generate.decode_step(step,text,{})
