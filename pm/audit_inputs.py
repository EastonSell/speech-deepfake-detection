"""Read-only, standard-library input preflight; not a training/evaluation run."""
import argparse
import json
from pathlib import Path
import wave


def inspect_audio(path, audio_root, expected_sample_rate):
    """Inspect PCM WAV bytes only; other formats remain explicitly unverified."""
    result = {'status': 'failed'}
    errors = []
    try:
        resolved = path.resolve()
        try:
            resolved.relative_to(audio_root)
        except ValueError:
            return result, ['audio_outside_directory']
        if not path.is_file():
            return result, ['missing_audio']
        if path.suffix.lower() != '.wav':
            return {'status': 'unverified', 'reason': 'decoder_not_available_in_stdlib'}, []
        with wave.open(str(path), 'rb') as audio:
            channels, width, rate, frames, _, _ = audio.getparams()
            result.update(sample_rate=rate, channels=channels, frames=frames,
                          sample_width_bytes=width)
            if rate != expected_sample_rate:
                errors.append('unexpected_sample_rate')
            if frames == 0:
                errors.append('empty_audio')
            # Stream rather than allocating an entire recording in memory.
            read_bytes = 0
            while True:
                chunk = audio.readframes(65536)
                if not chunk:
                    break
                read_bytes += len(chunk)
            if read_bytes != frames * channels * width:
                errors.append('truncated_wav')
        if not errors:
            result['status'] = 'checked'
    except (OSError, EOFError, wave.Error, ValueError, RuntimeError) as exc:
        errors.append('unreadable_wav')
        result['detail'] = str(exc)
    return result, errors


def audit(metadata, audio_dir, audio_ext='.wav', expected_sample_rate=16000):
    """Account for every physical protocol line, including rejected records.

    The existing loader consumes the first eight whitespace-separated fields,
    accepts extra fields, and maps labels beginning with 'bona' to real. Keep
    that grammar here without importing its optional torch/soundfile stack.
    """
    if audio_ext not in ('.wav', '.flac'):
        raise ValueError('audio_ext must be .wav or .flac')
    if expected_sample_rate <= 0:
        raise ValueError('expected_sample_rate must be positive')
    counts = dict(lines=0, blank=0, accepted=0, rejected=0,
                  audio_checked=0, audio_failed=0, audio_unverified=0)
    report = {'schema_version': 1, 'checks_passed': False,
              'training_readiness_established': False,
              'scope': 'protocol, paths, and PCM WAV structure only; no model or efficacy check',
              'expected_sample_rate': expected_sample_rate,
              'counts': counts, 'errors': [], 'rows': []}
    seen = set()
    try:
        audio_root = Path(audio_dir).resolve()
    except (OSError, RuntimeError, ValueError) as exc:
        audio_root = None
        report['errors'].append('audio_directory_unreadable')
        report['audio_directory_error_detail'] = str(exc)
    try:
        with Path(metadata).open('r', encoding='utf-8', errors='surrogateescape') as protocol:
            for line_number, raw in enumerate(protocol, 1):
                counts['lines'] += 1
                row = {'line': line_number, 'errors': []}
                report['rows'].append(row)
                try:
                    raw.encode('utf-8')  # Reject undecodable bytes, retaining line counts.
                    parts = raw.split()
                except UnicodeEncodeError:
                    row.update(status='rejected', errors=['invalid_utf8'])
                    counts['rejected'] += 1
                    continue
                if not parts:
                    row['status'] = 'blank'
                    counts['blank'] += 1
                    continue
                if len(parts) < 8:
                    row.update(status='rejected', errors=['too_few_fields'])
                    counts['rejected'] += 1
                    continue
                label = parts[5].lower()
                label = ('fake' if label in ('fake', 'spoof') else
                         'real' if label == 'real' or label.startswith('bona') else None)
                if label is None:
                    row.update(status='rejected', errors=['unknown_label'])
                    counts['rejected'] += 1
                    continue
                trial_id = parts[1]
                row.update(status='accepted', trial_id=trial_id, label=label)
                counts['accepted'] += 1
                if trial_id in seen:
                    row['errors'].append('duplicate_trial_id')
                seen.add(trial_id)
                if trial_id in ('.', '..') or any(c in trial_id for c in ('/', '\\', ':', '\0')):
                    row['errors'].append('unsafe_trial_id')
                    row['audio'] = {'status': 'failed'}
                elif audio_root is None:
                    row['audio'] = {'status': 'failed'}
                    row['errors'].append('audio_directory_unreadable')
                else:
                    row['audio'], errors = inspect_audio(
                        audio_root / (trial_id + audio_ext), audio_root, expected_sample_rate)
                    row['errors'].extend(errors)
                counts['audio_' + row['audio']['status']] += 1
    except OSError as exc:
        report['errors'].append('metadata_unreadable')
        report['metadata_error_detail'] = str(exc)
    if not counts['accepted']:
        report['errors'].append('no_accepted_records')
    report['checks_passed'] = not (report['errors'] or counts['rejected'] or
        counts['audio_failed'] or counts['audio_unverified'] or
        any(row['errors'] for row in report['rows']))
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metadata', required=True)
    parser.add_argument('--audio-dir', required=True)
    parser.add_argument('--audio-ext', choices=('.wav', '.flac'), default='.wav')
    parser.add_argument('--expected-sample-rate', type=int, default=16000,
                        help='audit expectation only; does not resample (default: 16000)')
    args = parser.parse_args(argv)
    if args.expected_sample_rate <= 0:
        parser.error('--expected-sample-rate must be positive')
    report = audit(args.metadata, args.audio_dir, args.audio_ext, args.expected_sample_rate)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report['checks_passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
