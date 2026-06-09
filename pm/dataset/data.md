# Dataset Notes

This project uses ASVspoof 2021 Logical Access audio for real-vs-fake speech detection.

The repo includes four small WAV files in `assets/example_audio/` and a matching `assets/trial_metadata.txt` file. These files are only for demos and quick checks.

Use the full ASVspoof files for training:

- Audio archive: https://zenodo.org/records/4837263
- Keys archive: https://www.asvspoof.org/asvspoof2021/LA-keys-full.tar.gz
- Challenge page: https://www.asvspoof.org/index2021.html

Expected paths after cloning on Talapas:

```text
pm/dataset/ASVspoof2021_LA_eval.tar.gz
pm/dataset/LA-keys-full.tar.gz
pm/dataset/ASVspoof2021_LA_eval/flac/
pm/dataset/ASVspoof2021_LA_eval/trial_metadata.txt
pm/model/weights/model.pt
```

The full archives and FLAC files are ignored by Git because they are too large for a normal GitHub repo.

The metadata file has eight whitespace-separated fields:

```text
speaker_id trial_id codec transmission attack label trim subset
LA_0001 LA_E_0000001 none loc_tx real real notrim eval
LA_0002 LA_E_0000002 alaw ita_tx A07 fake notrim eval
```

Only `trial_id` and `label` are required for training. The other fields are kept so results can later be checked by codec or transmission condition.
