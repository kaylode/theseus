import re
from pathlib import Path
import os.path as osp
import setuptools

## Check repo version
version = ''
with open('theseus/__init__.py') as f:
    version = re.search(r'^__version__\s*=\s*[\'"]([^\'"]*)[\'"]', f.read(), re.MULTILINE).group(1)

## Base requirements
with open("requirements.txt", 'r') as f:
    install_requires = f.read().splitlines()

## Check extra requirements recursively
extras_requires = {}
extra_req_files = list(Path('theseus').rglob("requirements.txt"))
all_extra_reqs = []
for req_file in extra_req_files:
    with open(req_file, 'r') as f:
        reqs = f.read().splitlines()
    pardir = osp.dirname(req_file).split('theseus/')[1].replace('/', '.')
    extras_requires.update({
        pardir: reqs,
    })
    all_extra_reqs.extend(reqs)

extras_requires['all'] = all_extra_reqs

setuptools.setup(
    name="theseus",
    author='kaylode',
    version=version,
    url='https://github.com/kaylode/theseus.py',
    license='MIT',
    description='A project contains templates and useful tools for Deep Learning',
    packages=setuptools.find_packages(),
    python_requires='>=3.7',
    install_requires=install_requires,
    extras_require=extras_requires
)