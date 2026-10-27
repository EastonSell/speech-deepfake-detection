"""Real-file preflight contracts; no ML dependencies or mocks."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import wave


class AuditInputsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.audio = self.root / 'audio'
        self.audio.mkdir()
        self.protocol = self.root / 'protocol.txt'

    def wav(self, name='a', rate=16000, frames=400):
        path = self.audio / (name + '.wav')
        with wave.open(str(path), 'wb') as f:
            f.setnchannels(1); f.setsampwidth(2); f.setframerate(rate)
            f.writeframes(b'\0\0' * frames)
        return path

    def run_audit(self, lines, ext='.wav'):
        self.protocol.write_text(lines, encoding='utf-8')
        p = subprocess.run([sys.executable, '-m', 'pm.audit_inputs',
            '--metadata', str(self.protocol), '--audio-dir', str(self.audio),
            '--audio-ext', ext], capture_output=True, text=True)
        self.assertIn(p.returncode, (0, 1), p.stderr)
        self.assertTrue(p.stdout.startswith('{'), p.stderr or p.stdout)
        return p.returncode, json.loads(p.stdout)

    def test_valid_wav_is_checked_without_claiming_training_readiness(self):
        self.wav()
        code, r = self.run_audit('s a none tx A spoof trim eval extra\n')
        self.assertEqual(code, 0)
        self.assertEqual(r['counts'], {'lines': 1, 'blank': 0, 'accepted': 1,
            'rejected': 0, 'audio_checked': 1, 'audio_failed': 0, 'audio_unverified': 0})
        self.assertEqual(r['rows'][0]['label'], 'fake')
        self.assertEqual(r['rows'][0]['audio']['sample_rate'], 16000)
        self.assertFalse(r['training_readiness_established'])

    def test_every_line_and_skipped_label_is_accounted_for(self):
        self.wav()
        code, r = self.run_audit('\ns a none tx A bonaFide trim eval\nshort row\ns b n t a other x y\n')
        self.assertEqual(code, 1)
        self.assertEqual(r['counts']['lines'], 4)
        self.assertEqual(r['counts']['blank'], 1)
        self.assertEqual(r['counts']['accepted'], 1)
        self.assertEqual(r['counts']['rejected'], 2)
        self.assertEqual([x['line'] for x in r['rows']], [1, 2, 3, 4])
        self.assertEqual(r['rows'][1]['label'], 'real')

    def test_duplicates_and_missing_audio_are_not_silently_dropped(self):
        code, r = self.run_audit('s a n t a real x y\ns a n t a fake x y\n')
        self.assertEqual(code, 1)
        self.assertEqual(r['counts']['accepted'], 2)
        self.assertEqual(r['counts']['audio_failed'], 2)
        self.assertIn('duplicate_trial_id', r['rows'][1]['errors'])
        self.assertIn('missing_audio', r['rows'][0]['errors'])

    def test_traversal_and_symlink_escape_are_rejected(self):
        external = self.root / 'external.wav'
        external.write_bytes(b'not audio')
        (self.audio / 'link.wav').symlink_to(external)
        code, r = self.run_audit('s ../external n t a real x y\ns link n t a real x y\n')
        self.assertEqual(code, 1)
        self.assertIn('unsafe_trial_id', r['rows'][0]['errors'])
        self.assertIn('audio_outside_directory', r['rows'][1]['errors'])

    def test_wrong_rate_empty_corrupt_and_truncated_audio_fail(self):
        self.wav('rate', rate=8000)
        self.wav('empty', frames=0)
        (self.audio / 'corrupt.wav').write_bytes(b'not wav')
        p = self.wav('truncated'); p.write_bytes(p.read_bytes()[:-20])
        code, r = self.run_audit(''.join('s %s n t a real x y\n' % n
            for n in ['rate', 'empty', 'corrupt', 'truncated']))
        self.assertEqual(code, 1)
        self.assertEqual(r['counts']['audio_failed'], 4)
        for row, err in zip(r['rows'], ['unexpected_sample_rate', 'empty_audio', 'unreadable_wav', 'truncated_wav']):
            self.assertIn(err, row['errors'])

    def test_flac_exists_but_remains_unverified(self):
        (self.audio / 'a.flac').write_bytes(b'not actual flac')
        code, r = self.run_audit('s a n t a real x y\n', '.flac')
        self.assertEqual(code, 1)
        self.assertEqual(r['counts']['audio_unverified'], 1)
        self.assertEqual(r['counts']['audio_checked'], 0)
        self.assertFalse(r['checks_passed'])

    def test_empty_protocol_fails(self):
        code, r = self.run_audit('\n')
        self.assertEqual(code, 1)
        self.assertIn('no_accepted_records', r['errors'])

    def test_cr_only_lines_match_loader_universal_newlines(self):
        self.wav('a'); self.wav('b')
        code, r = self.run_audit('s a n t a real x y\rs b n t a fake x y\r')
        self.assertEqual(code, 0)
        self.assertEqual(r['counts']['lines'], 2)
        self.assertEqual(r['counts']['audio_checked'], 2)

    def test_audio_root_symlink_loop_is_json_failure(self):
        self.audio.rmdir()
        self.audio.symlink_to(self.audio)
        code, r = self.run_audit('s a n t a real x y\n')
        self.assertEqual(code, 1)
        self.assertIn('audio_directory_unreadable', r['errors'])
        self.assertEqual(r['counts']['accepted'], 1)
        self.assertEqual(r['counts']['audio_failed'], 1)

    def test_invalid_utf8_does_not_erase_later_record_or_mutate_inputs(self):
        from pm.audit_inputs import audit
        audio = self.wav()
        before_audio = audio.read_bytes()
        before_protocol = b'bad \xff row\r\ns a n t a real x y\n'
        self.protocol.write_bytes(before_protocol)
        r = audit(self.protocol, self.audio)
        self.assertFalse(r['checks_passed'])
        self.assertEqual(r['counts']['lines'], 2)
        self.assertEqual(r['counts']['rejected'], 1)
        self.assertEqual(r['counts']['audio_checked'], 1)
        self.assertIn('invalid_utf8', r['rows'][0]['errors'])
        self.assertEqual(self.protocol.read_bytes(), before_protocol)
        self.assertEqual(audio.read_bytes(), before_audio)

    def test_invalid_expectation_is_usage_error(self):
        p = subprocess.run([sys.executable, '-m', 'pm.audit_inputs', '--metadata',
            str(self.protocol), '--audio-dir', str(self.audio),
            '--expected-sample-rate', '0'], capture_output=True, text=True)
        self.assertEqual(p.returncode, 2)
        self.assertIn('must be positive', p.stderr)

    def test_missing_protocol_is_json_failure(self):
        p = subprocess.run([sys.executable, '-m', 'pm.audit_inputs', '--metadata',
            str(self.protocol), '--audio-dir', str(self.audio)], capture_output=True, text=True)
        self.assertEqual(p.returncode, 1)
        self.assertTrue(p.stdout.startswith('{'), p.stderr)
        self.assertIn('metadata_unreadable', json.loads(p.stdout)['errors'])


if __name__ == '__main__':
    unittest.main()
