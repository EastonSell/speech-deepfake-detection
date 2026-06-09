# Dataset Notes

This project uses ASVspoof Logical Access speech data for real-vs-spoofed audio detection.

For the milestone, the repository includes four tiny WAV files in `assets/example_audio/` and an ASVspoof-style `assets/trial_metadata.txt`. These examples make the notebook and package runnable immediately after a fresh install.

For the full project, use official ASVspoof data:

- ASVspoof 2021 LA speech archive: https://zenodo.org/records/4837263
- ASVspoof 2021 LA keys/metadata: https://www.asvspoof.org/asvspoof2021/LA-keys-full.tar.gz
- ASVspoof challenge site: https://www.asvspoof.org/index2021.html

Phone-quality or low-quality audio comes from ASVspoof 2021 LA codec/transmission metadata such as `alaw`, `pstn`, `g722`, `ulaw`, `gsm`, and `opus`.

Download the eval archive into `pm/dataset/` (this is where the repo expects it):

```
pm/dataset/ASVspoof2021_LA_eval.tar.gz     


Resulting tree the training/eval code reads:

```
pm/dataset/ASVspoof2021_LA_eval/
├── flac/                       # (~181k utterances)
└── trial_metadata.txt          # speaker_id trial_id codec transmission attack label trim subset
```

## Protocol format

`read_protocol()` in `dataloader.py` reads eight whitespace-separated columns:

```
speaker_id  trial_id  codec  transmission  attack  label  trim  subset
LA_0001     LA_E_0000001  none   loc_tx     bonafide  bonafide  notrim  eval
LA_0002     LA_E_0000002  alaw   ita_tx     A07       spoof     notrim  eval
```

Only `trial_id` (to locate the audio file) and `label` (the target) are used for
training; the `codec` / `transmission` columns are retained so model performance can
be sliced by channel condition. Phone-quality / low-quality audio corresponds to
codec/transmission tags such as `alaw`, `pstn`, `g722`, `ulaw`, `gsm`, and `opus`.

## Where to find the data and weights on Talapas

- **Data:** `pm/dataset/ASVspoof2021_LA_eval.tar.gz` → extracted to
  `pm/dataset/ASVspoof2021_LA_eval/` (also mirrorable under
  `/projects/<PIRG>/<duckid>/...` if staged on shared scratch).
- **Trained weights:** `pm/model/weights/model.pt`, written by
  `pm/train_model.py` after training completes.
