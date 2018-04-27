import numpy as np
from Cython.Build import cythonize
from setuptools import Extension, setup

ext_modules = [
    Extension(
        'davisinteractive.third_party._mask',
        sources=[
            'davisinteractive/third_party/maskApi.c',
            'davisinteractive/third_party/_mask.pyx'
        ],
        include_dirs=[np.get_include(), 'davisinteractive/third_party'],
        extra_compile_args=['-Wno-cpp', '-Wno-unused-function', '-std=c99'],
    )
]

setup(
    python_requires='>=2.7, !=3.0.*, !=3.1.*, !=3.2.*, !=3.3.*, <4',
    install_requires=[
        'numpy>=1.12.1', 'scikit-learn>=0.18', 'scikit-image>=0.13.1',
        'networkx>=2.0', 'scipy>=1.0.0', 'pandas>=0.21.1', 'absl-py>=0.1.13',
        'Pillow>=4.1.1', 'mock;python_version<"3.0"',
        'pathlib2;python_version<"3.5"', 'six>=1.10.0'
    ],
    ext_modules=cythonize(ext_modules))
