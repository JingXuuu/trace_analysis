import json
import tempfile
import unittest
from pathlib import Path

from trace_analysis.formats import detect_format, load_trace, source_files
from trace_analysis.content_traces import RAG_FILES
from trace_analysis.reuse import source_key
from trace_analysis.web import AnalysisService


class ContentTraceTests(unittest.TestCase):
    def test_bailian_prefix_scoping_and_partial_lengths(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'bailian.jsonl'
            records = [dict(chat_id=i, input_length=length, hash_ids=ids)
                       for i, (length, ids) in enumerate([(20,[1,2]), (20,[1,2]),
                                                        (32,[3,2]), (32,[1,2])])]
            path.write_text(''.join(json.dumps(r)+'\n' for r in records))
            self.assertEqual(detect_format(path), 'bailian')
            (rows, sizes, count), _ = load_trace(path, 512)
            self.assertEqual(rows[0].hash_ids, rows[1].hash_ids)
            self.assertNotEqual(rows[0].hash_ids[-1], rows[2].hash_ids[-1])
            self.assertNotEqual(rows[0].hash_ids[-1], rows[3].hash_ids[-1])
            self.assertEqual(sum(sizes[h] for h in rows[0].hash_ids), 20)
            self.assertEqual(count, 8)
            records[0]['input_length'] = 100
            path.write_text(json.dumps(records[0])+'\n')
            with self.assertRaises(ValueError):
                load_trace(path, 16)

    def test_ragpulse_lookup_order_discovery_and_invalidation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in RAG_FILES:
                (root/name).write_text('')
            lookup = root/RAG_FILES[0]
            lookup.write_text(''.join(json.dumps(dict(sys_prompt_id=i, token_length=n))+'\n'
                                      for i,n in [(1,10),(2,20),(3,0),(4,30)]))
            path = root/'0_trace.jsonl'
            records = []
            for session, system in [('a',1),('b',1),('c',4)]:
                records.append(dict(session_id=session, input_length=30,
                    hash_ids=dict(sys_prompt=[system], passages_ids=[2], history=[3],
                                  web_search=[], user_input=[])))
            path.write_text(''.join(json.dumps(r)+'\n' for r in records))
            self.assertEqual(detect_format(path), 'ragpulse')
            (rows,sizes,_), _ = load_trace(path,512)
            self.assertEqual(rows[0].hash_ids, rows[1].hash_ids)
            self.assertNotEqual(rows[0].hash_ids[-1],rows[2].hash_ids[-1])
            self.assertEqual([sizes[h] for h in rows[0].hash_ids],[10,20])
            self.assertEqual(len(source_files(path)),6)
            self.assertEqual(len(AnalysisService([root],root/'outputs').traces()),1)
            key = source_key(path)
            lookup.write_text(lookup.read_text()+'\n')
            self.assertNotEqual(key,source_key(path))
            # Reordering components changes the prefix, even with the same IDs.
            record = records[0]
            record['hash_ids'] = dict(passages_ids=[2], sys_prompt=[1], history=[],
                                      web_search=[], user_input=[])
            path.write_text(json.dumps(record)+'\n')
            (reordered, reordered_sizes, _), _ = load_trace(path,512)
            self.assertEqual([reordered_sizes[h] for h in reordered[0].hash_ids],[20,10])
            lookup.unlink()
            self.assertEqual(AnalysisService([root],root/'outputs').traces(), [])
            with self.assertRaises(FileNotFoundError):
                load_trace(path,512)
