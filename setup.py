import numpy as np
from Cython.Build import cythonize
from setuptools import Extension, setup

ext_modules = [
    Extension(
        'davisinteractive.third_party.mask_api._mask',
        sources=[
            'davisinteractive/third_party/mask_api/maskApi.c',
            'davisinteractive/third_party/mask_api/_mask.pyx'
        ],
        include_dirs=[
            np.get_include(), 'davisinteractive/third_party/mask_api'
        ],
        extra_compile_args=['-Wno-cpp', '-Wno-unused-function', '-std=c99'],
    )
]

setup(
    python_requires='>=2.7, !=3.0.*, !=3.1.*, !=3.2.*, !=3.3.*, <4',
    setup_requires=['numpy', 'cython'],
    install_requires=[
        'Pillow>=4.1.1',
        'absl-py>=0.1.13',
        'networkx>=2.3',
        'numpy>=1.12.1',
        'opencv-python>=4.0.0.21',
        'pandas>=1.0.0',
        'pathlib2;python_version<"3.5"',
        'requests>=2.21.0',
        'scikit-image>=0.13.1',
        'scikit-learn==0.20.3',
        'scipy>=1.0.0',
        'six>=1.10.0',
    ],
    # test_require=['mock;python_version<"3.0"'],
    ext_modules=cythonize(ext_modules))
