"""Record-level split provenance for newly trained checkpoints."""
import hashlib
import json
import os
import tempfile
from pathlib import Path


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def validate_records(records):
    seen = set()
    for record in records:
        trial_id = record.get('trial_id')
        if not isinstance(trial_id, str) or not trial_id:
            raise ValueError('each record must have a nonempty trial_id')
        if trial_id in seen:
            raise ValueError('duplicate trial_id: ' + trial_id)
        seen.add(trial_id)


def build_manifest(records, train_indices, validation_indices, metadata_path, seed, val_frac, limit, expected_metadata_sha256=None):
    validate_records(records)
    train_indices, validation_indices = list(train_indices), list(validation_indices)
    all_indices = train_indices + validation_indices
    if any(type(i) is not int for i in all_indices) or sorted(all_indices) != list(range(len(records))):
        raise ValueError('split indices must cover every record exactly once')
    def entries(indices):
        return [{'index': i, 'record': dict(records[i])} for i in indices]
    metadata_sha256 = sha256_file(metadata_path)
    if expected_metadata_sha256 is not None and metadata_sha256 != expected_metadata_sha256:
        raise ValueError('metadata changed while loading the dataset')
    return {'version': 1, 'metadata_sha256': metadata_sha256,
            'seed': seed, 'val_frac': val_frac, 'limit': limit,
            'train': entries(train_indices), 'validation': entries(validation_indices)}


def _write_manifest(path, manifest):
    # The temporary file is in the output directory, so replace is atomic.
    with tempfile.NamedTemporaryFile('w', encoding='utf-8', dir=path.parent,
                                     prefix='.split-', delete=False) as stream:
        tmp = Path(stream.name)
        try:
            json.dump(manifest, stream, indent=2, sort_keys=True)
            stream.write('\n')
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
    try:
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def publish_checkpoint(out_path, manifest, save_weights):
    """Publish sidecar first; a failure to write it cannot replace weights.

    The caller must verify checkpoint_sha256 when using a sidecar. Orphan
    sidecars are possible if the process fails before checkpoint replacement.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=out_path.parent, prefix='weights-', delete=False) as stream:
        temp_path = Path(stream.name)
    try:
        save_weights(temp_path)
        digest = sha256_file(temp_path)
        payload = dict(manifest, checkpoint_sha256=digest)
        manifest_digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':')).encode('utf-8')).hexdigest()
        sidecar = out_path.with_name(out_path.name + '.split.' + digest + '.' + manifest_digest + '.json')
        _write_manifest(sidecar, payload)
        os.replace(temp_path, out_path)
        return sidecar
    finally:
        temp_path.unlink(missing_ok=True)
