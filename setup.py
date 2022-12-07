import re
import os.path as osp
import setuptools
from theseus.utilities.folder import find_file_recursively


## Check repo version
version = ''
with open('theseus/__init__.py') as f:
    version = re.search(r'^__version__\s*=\s*[\'"]([^\'"]*)[\'"]', f.read(), re.MULTILINE).group(1)

## Base requirements
with open("requirements.txt", 'r') as f:
    install_requires = f.read().splitlines()

## Check extra requirements recursively
extras_requires = {}
extra_req_files = find_file_recursively('theseus', filename="requirements.txt")
for req_file in extra_req_files:
    with open(req_file, 'r') as f:
        reqs = f.read().splitlines()
    pardir = osp.dirname(req_file).split('theseus/')[1].replace('/', '.')
    extras_requires.update({
        pardir: reqs,
    })

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