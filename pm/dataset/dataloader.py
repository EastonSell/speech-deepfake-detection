from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import DataLoader, Dataset

LABEL_TO_ID = {"fake": 0, "real": 1}


class ASVspoofAudioDataset(Dataset):
    def __init__(self, protocol_path=None, audio_dir=None, max_frames=96, audio_ext=".wav"):
        asset_dir = Path(__file__).resolve().parents[2] / "assets"
        self.protocol_path = Path(protocol_path) if protocol_path else asset_dir / "trial_metadata.txt"
        self.audio_dir = Path(audio_dir) if audio_dir else asset_dir / "example_audio"
        self.max_frames = max_frames
        self.audio_ext = audio_ext
        self.records = read_protocol(self.protocol_path)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]
        audio_path = self.audio_dir / f"{record['trial_id']}{self.audio_ext}"
        waveform, sample_rate = load_audio(audio_path)
        spectrogram = waveform_to_log_spectrogram(waveform, sample_rate)
        spectrogram = pad_or_crop(spectrogram, self.max_frames)
        label = LABEL_TO_ID[record["label"]]
        return torch.tensor(spectrogram[None, :, :], dtype=torch.float32), label


def get_data_loaders(base_path=None, name="example", batch_size=2):
    if name == "example":
        dataset = ASVspoofAudioDataset()
    else:
        base_path = Path(base_path)
        dataset = ASVspoofAudioDataset(
            protocol_path=base_path / "trial_metadata.txt",
            audio_dir=base_path / "flac",
            audio_ext=".flac",
        )

    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def read_protocol(protocol_path):
    records = []
    with Path(protocol_path).open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 8:
                continue
            speaker_id, trial_id, codec, transmission, attack, label, trim, subset = parts[:8]
            label = normalize_label(label)
            if label is None:
                continue
            records.append(
                {
                    "speaker_id": speaker_id,
                    "trial_id": trial_id,
                    "codec": codec,
                    "transmission": transmission,
                    "attack": attack,
                    "label": label,
                    "trim": trim,
                    "subset": subset,
                }
            )
    return records


def normalize_label(label):
    label = label.lower()
    if label in {"fake", "spoof"}:
        return "fake"
    if label == "real" or label.startswith("bona"):
        return "real"
    return None


def load_audio(path):
    waveform, sample_rate = sf.read(str(path), dtype="float32")
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1)
    return waveform, sample_rate


FRAME_SAMPLES = 400
HOP_SAMPLES = 160


def waveform_to_log_spectrogram(waveform, sample_rate=None):
    frame_length = FRAME_SAMPLES
    hop_length = HOP_SAMPLES
    if len(waveform) < frame_length:
        waveform = np.pad(waveform, (0, frame_length - len(waveform)))
    frames = []
    for start in range(0, len(waveform) - frame_length + 1, hop_length):
        frames.append(waveform[start : start + frame_length])
    frames = np.stack(frames)
    spectrum = np.fft.rfft(frames * np.hanning(frame_length), axis=1)
    return np.log(np.abs(spectrum) ** 2 + 1e-8).astype(np.float32)


def pad_or_crop(spectrogram, max_frames):
    output = np.zeros((max_frames, spectrogram.shape[1]), dtype=np.float32)
    frames = min(max_frames, spectrogram.shape[0])
    output[:frames] = spectrogram[:frames]
    return (output - output.mean()) / (output.std() + 1e-8)
