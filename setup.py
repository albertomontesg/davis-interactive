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
        'numpy>=1.12.1',
        'scikit-learn>=0.18',
        'scikit-image>=0.13.1',
        'networkx>=2.0',
        'scipy>=1.0.0',
        'pandas>=0.21.1',
        'absl-py>=0.1.13',
        'Pillow>=4.1.1',
        'pathlib2;python_version<"3.5"',
        'six>=1.10.0',
    ],
    # test_require=['mock;python_version<"3.0"'],
    ext_modules=cythonize(ext_modules))
