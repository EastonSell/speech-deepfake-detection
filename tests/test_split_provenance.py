import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pm.split_provenance import build_manifest, publish_checkpoint

RECORDS = [dict(trial_id='a', label='fake', speaker_id='s1'),
           dict(trial_id='b', label='real', speaker_id='s1'),
           dict(trial_id='c', label='fake', speaker_id='s2')]

class SplitProvenanceTests(unittest.TestCase):
    def test_records_are_exact_and_bound_to_metadata(self):
        with tempfile.TemporaryDirectory() as d:
            metadata = Path(d)/'protocol'; metadata.write_bytes(b'protocol')
            m = build_manifest(RECORDS, [2, 0], [1], metadata, 7, .34, None)
            self.assertEqual([x['record'] for x in m['train']], [RECORDS[2], RECORDS[0]])
            self.assertEqual([x['index'] for x in m['validation']], [1])
            self.assertEqual(m['metadata_sha256'], hashlib.sha256(b'protocol').hexdigest())

    def test_duplicate_trial_ids_rejected_even_with_different_labels(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'p'; p.write_text('x')
            with self.assertRaisesRegex(ValueError, 'duplicate trial_id'):
                build_manifest([RECORDS[0], dict(RECORDS[0],label='real')], [0], [1], p, 0, .5, None)

    def test_partition_must_be_complete_and_disjoint(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'p'; p.write_text('x')
            for train, val in [([0,1],[1,2]), ([0],[1]), ([0,1],[2,3])]:
                with self.subTest(train=train,val=val), self.assertRaises(ValueError):
                    build_manifest(RECORDS, train, val, p, 0, .5, None)

    def test_sidecar_is_published_before_checkpoint_and_digest_matches(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d)/'model.pt'
            path=publish_checkpoint(out, {'version': 1}, lambda p: Path(p).write_bytes(b'weights'))
            m=json.loads(path.read_text())
            digest=hashlib.sha256(out.read_bytes()).hexdigest()
            self.assertEqual(m['checkpoint_sha256'],digest)
            self.assertIn(digest, path.name)

    def test_sidecar_write_failure_does_not_replace_checkpoint(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d)/'model.pt'; out.write_bytes(b'old')
            with patch('pm.split_provenance._write_manifest', side_effect=OSError('disk full')):
                with self.assertRaises(OSError):
                    publish_checkpoint(out, {}, lambda p: Path(p).write_bytes(b'new'))
            self.assertEqual(out.read_bytes(),b'old')

    def test_same_weights_distinct_splits_keep_both_manifests(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / 'model.pt'
            first = publish_checkpoint(out, {'train': [1]}, lambda p: Path(p).write_bytes(b'weights'))
            second = publish_checkpoint(out, {'train': [2]}, lambda p: Path(p).write_bytes(b'weights'))
            self.assertNotEqual(first, second)
            self.assertEqual(json.loads(first.read_text())['train'], [1])
            self.assertEqual(json.loads(second.read_text())['train'], [2])

    def test_metadata_changed_since_read_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'protocol'; p.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'metadata changed'):
                build_manifest(RECORDS, [0, 1], [2], p, 0, .2, None, expected_metadata_sha256=hashlib.sha256(b'original').hexdigest())

    def test_failed_checkpoint_replace_preserves_original_manifest(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / 'model.pt'
            first = publish_checkpoint(out, {'train': [1]}, lambda p: Path(p).write_bytes(b'weights'))
            original = first.read_bytes()
            import pm.split_provenance as provenance
            real_replace = provenance.os.replace
            def replace(src, dst):
                if Path(dst) == out: raise OSError('replace failed')
                return real_replace(src, dst)
            with patch('pm.split_provenance.os.replace', side_effect=replace):
                with self.assertRaises(OSError):
                    publish_checkpoint(out, {'train': [2]}, lambda p: Path(p).write_bytes(b'weights'))
            self.assertEqual(first.read_bytes(), original)
