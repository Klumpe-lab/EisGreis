from __future__ import annotations

from typing import Optional

import numpy as np
import numpy.typing as npt
import scipy.fft as fft


def extract_voxel_size(voxel_size: np.ndarray) -> float:
    valid = voxel_size[np.isfinite(voxel_size) & (voxel_size > 0)]
    if valid.size == 0:
        return 1.0
    return float(valid.mean())


def resolution_bounds(shape: tuple[int, ...], voxel_size_angstrom: float) -> tuple[float, float]:
    nyquist_resolution = 2.0 * voxel_size_angstrom
    max_dimension = max(shape) if shape else 1
    lowest_frequency = 1.0 / (max_dimension * voxel_size_angstrom)
    lowest_resolution = 1.0 / lowest_frequency
    return float(max(nyquist_resolution, 0.1)), float(max(lowest_resolution, nyquist_resolution + 0.5))


def frequency_grid(shape: tuple[int, ...], voxel_size_angstrom: float) -> np.ndarray:
    axes = np.meshgrid(
        *[fft.fftshift(np.fft.fftfreq(size, d=voxel_size_angstrom)) for size in shape],
        indexing="ij",
    )
    return np.sqrt(sum(axis**2 for axis in axes)).astype(np.float32, copy=False)


def lowpass_mask(
    frequency_grid_data: np.ndarray,
    resolution_angstrom: float,
    filter_type: str,
) -> np.ndarray:
    cutoff_frequency = 1.0 / max(resolution_angstrom, 1e-3)
    ratio = frequency_grid_data / max(cutoff_frequency, 1e-6)

    if filter_type == "hard":
        return (frequency_grid_data <= cutoff_frequency).astype(np.float32, copy=False)

    if filter_type == "gaussian":
        return np.exp(-(ratio**2) / 2.0).astype(np.float32, copy=False)

    if filter_type == "butterworth":
        order = 2
        return (1.0 / (1.0 + ratio ** (2 * order))).astype(np.float32, copy=False)

    if filter_type == "cosine":
        start = 0.8 * cutoff_frequency
        stop = cutoff_frequency
        mask = np.ones_like(frequency_grid_data, dtype=np.float32)
        outer = frequency_grid_data >= stop
        transition = (frequency_grid_data > start) & (frequency_grid_data < stop)
        mask[outer] = 0.0
        transition_ratio = (frequency_grid_data[transition] - start) / max(stop - start, 1e-6)
        mask[transition] = 0.5 * (1.0 + np.cos(np.pi * transition_ratio))
        return mask

    return np.exp(-(ratio**2) / 2.0).astype(np.float32, copy=False)


def apply_lowpass(
    data: np.ndarray,
    frequency_grid_data: np.ndarray,
    resolution_angstrom: float,
    filter_type: str,
) -> np.ndarray:
    spectrum = fft.fftshift(fft.fftn(data))
    mask = lowpass_mask(frequency_grid_data, resolution_angstrom, filter_type)
    filtered = fft.ifftn(fft.ifftshift(spectrum * mask))
    return np.asarray(np.real(filtered), dtype=np.float32)


def gaussian_bandpass_mask(
    lp_filter_resolution_ang: float,
    hp_filter_resolution_ang: float,
    pixel_size: float,
    input_shape: tuple[int, ...],
) -> np.ndarray:
    new_shape = np.max(input_shape)
    box_range = np.arange(-new_shape // 2 + 1, 1 + new_shape // 2)
    mesh_dist = np.sum(
        np.array(
            np.meshgrid(*([box_range] * len(input_shape)), sparse=True),
            dtype=object,
        )
        ** 2,
        axis=0,
    )

    mask = np.ones_like(mesh_dist, dtype=np.float32)
    if lp_filter_resolution_ang != 0:
        lp_filter_frequency = pixel_size / lp_filter_resolution_ang
        lp_radius = max(new_shape * lp_filter_frequency, 1e-6)
        mask *= np.exp(-mesh_dist / (2.0 * lp_radius**2)).astype(np.float32, copy=False)

    if hp_filter_resolution_ang != 0:
        hp_filter_frequency = pixel_size / hp_filter_resolution_ang
        hp_radius = max(new_shape * hp_filter_frequency, 1e-6)
        mask *= (1.0 - np.exp(-mesh_dist / (2.0 * hp_radius**2))).astype(np.float32, copy=False)

    return mask


def gaussian_bandpass_filter(
    input_data: npt.ArrayLike,
    lp_filter_resolution_ang: float,
    hp_filter_resolution_ang: float,
    pixel_size: float,
    log=print,
):
    if lp_filter_resolution_ang >= hp_filter_resolution_ang and hp_filter_resolution_ang != 0:
        log(f"{lp_filter_resolution_ang} cannot be greater than {hp_filter_resolution_ang}")
        return None

    input_data = np.asarray(input_data)
    mask = gaussian_bandpass_mask(
        lp_filter_resolution_ang=lp_filter_resolution_ang,
        hp_filter_resolution_ang=hp_filter_resolution_ang,
        pixel_size=pixel_size,
        input_shape=input_data.shape,
    )

    pad_list = []
    for shape_i in input_data.shape:
        diff_i = np.max(input_data.shape) - shape_i
        pad_list.append([diff_i // 2, diff_i // 2 + diff_i % 2])

    pad_image = np.pad(input_data, pad_list, "symmetric")
    pad_image = fft.fftn(pad_image, workers=-1)
    pad_image = fft.fftshift(pad_image)
    pad_image *= mask
    pad_image = fft.ifftshift(pad_image)
    pad_image = fft.ifftn(pad_image, workers=-1)

    unpad_list = []
    for pad_i, shape_i in zip(pad_list, input_data.shape):
        unpad_list.append(np.s_[pad_i[0] : pad_i[0] + shape_i])
    filtered = pad_image.real[tuple(unpad_list)]
    return filtered, mask
