"""Default preprocessing parameters.

Every value here mirrors the training pipeline of
``tetra-tools/RS-JYC-HK-T1wMNIBrainAgeRegression`` (``generate_ptfiles.py``),
with all data augmentation removed. A model bundle's ``manifest.json`` can
override them, so a future model trained with different settings does not
need a code change.
"""

# Model input grid (tio.CropOrPad target in generate_ptfiles.py)
TARGET_SHAPE = (160, 192, 160)

# tio.RescaleIntensity(out_min_max=(0, 1), percentiles=(0.5, 99.5))
RESCALE_OUT_MIN_MAX = (0.0, 1.0)
RESCALE_PERCENTILES = (0.5, 99.5)

# brain = ScalarImage('ravg152T1_1mm_defaced.nii').data > .1
MASK_THRESHOLD = 0.1

# Training images were "space-avg152T1_desc-biascorr": N4-corrected, then
# rigid + affine registered to avg152T1 1 mm (no non-linear warp).
REGISTRATION_TRANSFORM = "antsRegistrationSyNQuick[a]"
REGISTRATION_RANDOM_SEED = 123

# N4 parameters (same as predict_image_sfcn.py in the source repo)
N4_SHRINK_FACTOR = 4
N4_CONVERGENCE = {"iters": [50, 50, 50, 50], "tol": 0.0}
N4_SPLINE_PARAM = 100

# Training tensors were stored as float16 in the .bin cache files and the
# network was trained with 16-mixed precision. Rounding the preprocessed
# tensor through float16 reproduces exactly what the network saw.
FP16_ROUNDTRIP = True

# Names used inside a bundle
MANIFEST_NAME = "manifest.json"
BUNDLE_FORMAT_VERSION = 1
ONNX_INPUT_NAME = "t1"
ONNX_OUTPUT_NAME = "age"

# Where the Docker image looks for the bundle by default
DEFAULT_BUNDLE_ENV = "BRAINAGE_BUNDLE"
DEFAULT_BUNDLE_PATH = "/opt/brainage/model"
