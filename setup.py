import os
import re
from typing import Dict, List

from setuptools import find_packages, setup


def read_file(filename: str) -> str:
    with open(os.path.join(os.path.dirname(__file__), filename)) as f:
        return f.read().strip()


def read_version() -> str:
    regexp = re.compile(r"^__VERSION__\W*=\W*'([\d.abrc]+)'")

    with open(
        os.path.join(
            os.path.dirname(__file__), 'pytorch_optimizer', '__init__.py'
        )
    ) as f:
        for line in f:
            match = regexp.match(line)
            if match is not None:
                return match.group(1)

    raise RuntimeError('Cannot find version in pytorch_optimizer/__init__.py')


INSTALL_REQUIRES: List[str] = ['torch>=1.5.0']


CLASSIFIERS: List[str] = [
    'License :: OSI Approved :: Apache Software License',
    'Intended Audience :: Developers',
    'Intended Audience :: Science/Research',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.7',
    'Programming Language :: Python :: 3.8',
    'Operating System :: OS Independent',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
]

KEYWORDS: List[str] = sorted(
    [
        'pytorch-optimizer',
        'pytorch',
        'adamp',
        'sgdp',
        'madgrad',
        'ranger',
        'ranger21',
        'agc',
        'gc',
        'chebyshev_schedule',
        'lookahead',
        'radam',
        'adabound',
        'adahessian',
        'adabelief',
    ]
)

PROJECT_URLS: Dict[str, str] = {
    'Website': 'https://github.com/kozistr/pytorch_optimizer',
    'Issues': 'https://github.com/kozistr/pytorch_optimizer/issues',
}

setup(
    name='pytorch-optimizer',
    version=read_version(),
    description='pytorch-optimizer',
    long_description=read_file('README.md'),
    long_description_content_type='text/markdown',
    classifiers=CLASSIFIERS,
    platforms=['Linux', 'Windows'],
    author='kozistr',
    author_email='kozistr@gmail.com',
    url='https://github.com/kozistr/pytorch_optimizer',
    download_url='https://pypi.org/project/pytorch-optimizer/',
    license='Apache 2',
    packages=find_packages(),
    install_requires=INSTALL_REQUIRES,
    keywords=KEYWORDS,
    zip_safe=True,
    include_package_data=True,
    project_urls=PROJECT_URLS,
    python_requires='>=3.7.0',
)
