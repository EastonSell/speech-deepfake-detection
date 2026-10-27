# Read-only input preflight (no installation required)

From a source checkout, check metadata and PCM WAV files before training:

```sh
python -m pm.audit_inputs \
  --metadata assets/trial_metadata.txt \
  --audio-dir assets/example_audio > /tmp/speech-input-audit.json
```

This command uses only Python's standard library. It does not import PyTorch,
load checkpoints, train a model, download data, repair files, or change inputs.
The four bundled examples are only a plumbing check, **not a representative
speech evaluation**. This tool adds no detector-accuracy result.

Exit status is **0** when the implemented checks pass, **1** when there are
errors or unverified audio, and **2** for invalid command arguments. JSON goes
to stdout, including ordinary metadata-open failures. Each physical metadata
line has a row number and status, so malformed records are not silently dropped.
Blank lines are counted but are not errors. The counts obey:

- `lines = blank + accepted + rejected`
- `accepted = audio_checked + audio_failed + audio_unverified`

`accepted` means accepted by the current loader's metadata grammar, **not** safe
or ready for training. Duplicates and unsafe IDs still count as accepted records
but carry errors; no failing row is silently removed from the denominator.
An empty protocol, unknown labels, or fewer than eight fields fails preflight.
Extra fields are accepted, matching the loader. Labels `fake` and `spoof` map
to fake; `real` and labels starting with `bona` map to real, case-insensitively.
This intentionally preserves the loader's broad `bona` prefix behavior.

## What is checked

- Duplicate trial IDs, even when labels differ.
- Unsafe trial IDs and audio paths resolving outside the audio directory.
- Audio file presence.
- PCM WAV readability using `wave`, nonzero declared frame count, and whether
  all declared frame bytes can be read (including truncated payload detection).
- Sample rate against 16,000 Hz by default. The loader currently uses fixed
  400-sample frames and 160-sample hops and does **not** resample based on the
  supplied rate. A mismatched rate therefore changes the physical time and
  frequency scale. `--expected-sample-rate` changes the audit expectation only;
  it neither changes training preprocessing nor establishes compatibility with
  any checkpoint. Choose it from a known preprocessing contract, not to silence
  an error.

## Important boundaries

For the full FLAC dataset, run with `--audio-ext .flac`. Metadata, paths and
presence are still checked, but **all existing FLAC files remain unverified**
and the command exits 1: the standard library has no FLAC decoder. This is not
an assertion that those files are corrupt. Verify them separately with the
actual training decoder in its existing environment. Missing FLAC files are
reported as failures, not merely unverified.

Non-PCM WAV encodings supported by `soundfile` may be rejected by Python's
`wave` reader. This is a deliberately conservative PCM check, not a replacement
for decoding through the training pipeline. A readable WAV header/payload is
not proof of useful speech, meaningful labels, sufficient duration, non-silent
content, absence of NaNs in other encodings, or acceptable model input.

`training_readiness_established` is always false. Passing these checks does not
verify dependencies, CPU/GPU operation, model compatibility, class balance,
speaker/recording independence, duplicates under different IDs, checkpoint
provenance, split membership, or model quality. No scientific inference follows
from the audit. Files can change after or during inspection; this is not an
atomic snapshot or protection against concurrent path substitution. Run only
on ordinary local files in a trusted, stable workspace. Reports include trial
IDs and may include local paths in error details; treat them as private.

## Tests

```sh
python -m unittest discover -s tests -v
```

The new tests use generated PCM WAVs, invalid/truncated files, metadata edge
cases, and subprocess exit/JSON checks. They do not exercise ML dependencies.
