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
        "albumentations>=1.1.0",
        "torchvision",
        "tqdm",
        "timm>=0.5.4",
        "pyyaml>=5.1",
        "webcolors",
        "omegaconf",
        "gdown>=4.4.0",
        "grad-cam",
        "tabulate",
        "segmentation-models-pytorch",
        "opencv-python-headless==4.2.0.32" 
        
    ],
)