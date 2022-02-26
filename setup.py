import setuptools

setuptools.setup(
    name="theseus",
    version='0.0.1',
    packages=setuptools.find_packages(),
    python_requires='>=3.7',
    install_requires=[
        "numpy",
        "torch",
        "tensorboard",
        "albumentations",
        "torchvision",
        "tqdm",
        "timm",
        "pyyaml>=5.1",
        "webcolors",
        "omegaconf",
        "gdown==3.13.0",
        "grad-cam",
        "tabulate",
        "segmentation-models-pytorch"
    ],
)