import setuptools

with open("requirements.txt", 'r') as f:
    reqs = f.read().splitlines()

setuptools.setup(
    name="theseus",
    version='0.0.1',
    packages=setuptools.find_packages(),
    python_requires='>=3.7',
    install_requires=reqs,
)