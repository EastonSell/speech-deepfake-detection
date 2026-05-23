from pathlib import Path
import wave

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


LABEL_TO_ID = {"spoof": 0, "bonafide": 1}


class ASVspoofAudioDataset(Dataset):
    def __init__(self, protocol_path=None, audio_dir=None, max_frames=96):
        asset_dir = Path(__file__).resolve().parents[2] / "assets"
        self.protocol_path = Path(protocol_path) if protocol_path else asset_dir / "trial_metadata.txt"
        self.audio_dir = Path(audio_dir) if audio_dir else asset_dir / "example_audio"
        self.max_frames = max_frames
        self.records = read_protocol(self.protocol_path)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]
        audio_path = self.audio_dir / f"{record['trial_id']}.wav"
        waveform, sample_rate = load_wav(audio_path)
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
        )

    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def read_protocol(protocol_path):
    records = []
    with Path(protocol_path).open("r", encoding="utf-8") as f:
        for line in f:
            speaker_id, trial_id, codec, transmission, attack, label, trim, subset = line.split()
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


def load_wav(path):
    with wave.open(str(path), "rb") as f:
        sample_rate = f.getframerate()
        waveform = np.frombuffer(f.readframes(f.getnframes()), dtype="<i2")
    return waveform.astype(np.float32) / 32768.0, sample_rate


def waveform_to_log_spectrogram(waveform, sample_rate):
    frame_length = int(sample_rate * 0.025)
    hop_length = int(sample_rate * 0.010)
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
