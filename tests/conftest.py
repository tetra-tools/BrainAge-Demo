"""Synthetic fixtures: no real MRI data or trained weights are needed to run the tests.

* template / mask on the MNI 1 mm grid (182 x 218 x 182), like avg152T1
* a "raw" subject scan: rotated, shifted, scaled, on another grid, with a bias field and noise
* landmarks trained with torchio on these images
* a randomly initialised SFCN saved as a Lightning-style checkpoint
"""

from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
import torch

MNI_SHAPE = (182, 218, 182)
MNI_AFFINE = np.array([[-1, 0, 0, 90], [0, 1, 0, -126], [0, 0, 1, -72], [0, 0, 0, 1]], dtype=float)


def _phantom(shape, affine, rng=None, transform=np.eye(4)):
    """Brain-like phantom defined in world (mm) coordinates, sampled on (shape, affine).

    ``transform`` maps subject world coords -> template world coords (rigid/affine misalignment).
    """
    idx = np.stack(np.meshgrid(*[np.arange(s) for s in shape], indexing="ij"), -1).reshape(-1, 3)
    world = idx @ affine[:3, :3].T + affine[:3, 3]
    world = world @ transform[:3, :3].T + transform[:3, 3]
    x, y, z = world[:, 0], world[:, 1], world[:, 2] - 10  # brain centred ~ (0, -18, 8)
    y = y + 18
    head = (x / 72) ** 2 + (y / 90) ** 2 + (z / 70) ** 2
    brain = (x / 62) ** 2 + (y / 80) ** 2 + (z / 58) ** 2
    wm = (x / 45) ** 2 + (y / 60) ** 2 + (z / 40) ** 2
    vent_l = ((x - 10) / 6) ** 2 + ((y + 5) / 25) ** 2 + ((z - 5) / 10) ** 2
    vent_r = ((x + 10) / 6) ** 2 + ((y + 5) / 25) ** 2 + ((z - 5) / 10) ** 2
    lobe = ((x - 25) / 15) ** 2 + ((y - 40) / 15) ** 2 + ((z - 20) / 12) ** 2  # asymmetric feature
    img = np.zeros(len(idx), dtype=np.float32)
    img[head < 1] = 300            # scalp
    img[brain < 1] = 600           # grey matter
    img[wm < 1] = 900              # white matter
    img[lobe < 1] = 450
    img[(vent_l < 1) | (vent_r < 1)] = 150  # CSF
    img = img.reshape(shape)
    if rng is not None:
        img += rng.normal(0, 15, shape).astype(np.float32) * (img > 0)
    return img


@pytest.fixture(scope="session")
def assets(tmp_path_factory) -> dict:
    import torchio as tio
    from brainage.model import SFCN

    d: Path = tmp_path_factory.mktemp("assets")
    rng = np.random.default_rng(42)

    # template + mask (mask source is a float image thresholded at 0.1, like ravg152T1_1mm_defaced)
    tpl = _phantom(MNI_SHAPE, MNI_AFFINE)
    nib.save(nib.Nifti1Image(tpl, MNI_AFFINE), d / "template.nii.gz")
    brain_only = np.where(tpl >= 150, tpl / tpl.max(), 0).astype(np.float32)
    brain_only[tpl == 300] = 0
    nib.save(nib.Nifti1Image(brain_only, MNI_AFFINE), d / "mask_defaced.nii")

    # subject already in template space (like the training "space-avg152T1_desc-biascorr" files)
    in_space = _phantom(MNI_SHAPE, MNI_AFFINE, rng)
    nib.save(nib.Nifti1Image(in_space, MNI_AFFINE), d / "sub-01_space-avg152T1_desc-biascorr_T1w.nii.gz")

    # raw subject: 1.1 mm, RAS axes, misaligned by rotation + translation + scaling, bias field
    angle = np.deg2rad(7)
    rot = np.array([[1, 0, 0], [0, np.cos(angle), -np.sin(angle)], [0, np.sin(angle), np.cos(angle)]])
    misalign = np.eye(4)
    misalign[:3, :3] = rot * 1.04
    misalign[:3, 3] = [4.0, -6.0, 3.0]
    raw_shape = (176, 200, 170)
    raw_affine = np.diag([1.1, 1.1, 1.1, 1.0])
    raw_affine[:3, 3] = -np.array(raw_shape) * 1.1 / 2 + np.array([0, -10, 5])
    raw = _phantom(raw_shape, raw_affine, rng, transform=misalign)
    gx = np.linspace(0.8, 1.25, raw_shape[0])[:, None, None]
    gz = np.linspace(1.1, 0.9, raw_shape[2])[None, None, :]
    raw = (raw * gx * gz).astype(np.float32)
    nib.save(nib.Nifti1Image(raw, raw_affine), d / "sub-02_T1w.nii.gz")

    # landmarks, trained exactly like generate_ptfiles.generate_landmarks
    landmarks = tio.HistogramStandardization.train([str(d / "template.nii.gz"),
                                                    str(d / "sub-01_space-avg152T1_desc-biascorr_T1w.nii.gz")])
    np.save(d / "t1_landmarks.npy", landmarks)

    # Lightning-style checkpoint with random weights (small channel number to stay fast)
    torch.manual_seed(0)
    model = SFCN(channel_number=8, dropout=0.4)
    with torch.no_grad():
        model.classifier[-1].bias.fill_(50.0)  # make outputs look like ages
    ckpt = {"state_dict": model.state_dict(),
            "hyper_parameters": {"channel_number": 8, "learning_rate": 0.04, "dropout": 0.4,
                                 "optimizer": "sgd", "loss": "mae"},
            "pytorch-lightning_version": "2.5.2", "epoch": 10}
    torch.save(ckpt, d / "model_best_score.ckpt")

    return {k: d / v for k, v in {
        "template": "template.nii.gz", "mask": "mask_defaced.nii", "landmarks": "t1_landmarks.npy",
        "checkpoint": "model_best_score.ckpt", "in_space": "sub-01_space-avg152T1_desc-biascorr_T1w.nii.gz",
        "raw": "sub-02_T1w.nii.gz"}.items()} | {"dir": d}


@pytest.fixture(scope="session")
def bundle_dir(assets, tmp_path_factory) -> Path:
    from brainage.export import export_bundle

    out = tmp_path_factory.mktemp("bundles") / "brainage-sfcn-0.0.1"
    export_bundle(assets["checkpoint"], assets["landmarks"], assets["template"], assets["mask"],
                  out, version="0.0.1", notes="synthetic test bundle")
    return out
