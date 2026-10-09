from __future__ import annotations

import torch
from torch_dct import dct as torch_dct


def trajectory_to_frequency(
    trajectory: torch.Tensor,
    num_freqs: int = 256,
    fft_type: str = "rfft",
) -> torch.Tensor:
    if trajectory.ndim != 2:
        raise ValueError(
            f"Expected trajectory with shape [T, D], got {tuple(trajectory.shape)}."
        )

    if fft_type not in {"rfft", "dct"}:
        raise ValueError(f"Invalid FFT type: {fft_type}. Expected 'rfft' or 'dct'.")

    signal = trajectory.to(torch.float32)
    if fft_type == "rfft":
        spectrum = torch.fft.rfft(signal, dim=0, norm="ortho")
    else:
        spectrum = torch_dct(signal.transpose(0, 1), norm="ortho").transpose(0, 1)

    spectrum = spectrum[:num_freqs]
    if spectrum.shape[0] < num_freqs:
        pad = torch.zeros(
            num_freqs - spectrum.shape[0],
            spectrum.shape[1],
            dtype=spectrum.dtype,
            device=spectrum.device,
        )
        spectrum = torch.cat([spectrum, pad], dim=0)

    if fft_type == "rfft":
        return torch.cat([spectrum.real, spectrum.imag], dim=-1)

    return spectrum
