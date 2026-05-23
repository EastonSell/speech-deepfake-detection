# Speech Deepfake Detection

DSCI 410L project milestone repository.

## Project Overview

This project investigates speech deepfake detection. The goal is to classify an audio clip as either bonafide human speech or spoofed/generated speech. The model input is unstructured speech audio converted into log-spectrogram features.

## Data Overview

The full project will use ASVspoof Logical Access speech data.

- ASVspoof 2021 LA speech archive: https://zenodo.org/records/4837263
- ASVspoof 2021 LA keys/metadata: https://www.asvspoof.org/asvspoof2021/LA-keys-full.tar.gz
- ASVspoof challenge site: https://www.asvspoof.org/index2021.html


Low-quality and phone-quality audio will come from ASVspoof 2021 LA codec/transmission conditions. The metadata fields in this repo make it possible to compare model performance across these conditions.

![Train Batch](./assets/train_batch.png)

## Methods Overview

The package includes:

- `pm.dataset.dataloader`: ASVspoof protocol parser, WAV/FLAC audio loader, and PyTorch `DataLoader`
- `pm.model`: a small CNN for log-spectrogram inputs
- `pm.train_model`: a training loop for checking that the project is trainable

## Results

In Progress

## Conclusion

In Progress

## Installation

```bash
pip install .
```

## Package Use
