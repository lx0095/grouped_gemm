import os
from pathlib import Path

from setuptools import find_packages, setup

try:
    import torch
    from torch.utils.cpp_extension import BuildExtension
except ModuleNotFoundError as e:
    raise ModuleNotFoundError("No module named 'torch'. `torch` is required to install `grouped_gemm`.",) from e

cwd = Path(os.path.dirname(os.path.abspath(__file__)))

extra_deps = {}

extra_deps['dev'] = [
    'absl-py',
]

extra_deps['all'] = set(dep for deps in extra_deps.values() for dep in deps)

setup(
    name="grouped_gemm",
    version="0.3.0",
    author="Trevor Gale",
    author_email="tgale@stanford.edu",
    description="Grouped GEMM",
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url="https://github.com/tgale06/grouped_gemm",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: Unix",
    ],
    packages=find_packages(),
    cmdclass={"build_ext": BuildExtension},
    extras_require=extra_deps,
)
