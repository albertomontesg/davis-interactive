from setuptools import setup

setup(
    name='davisinteractive',
    version='0.0.1',
    description=
    'Evaluation tools for DAVIS interactive segmentation challenge.',
    url='http://github.com/albertomontesg/davis-interactive',
    author='Alberto Montes',
    author_email='al.montes.gomez@gmail.com',
    license='MIT',
    packages=['davisinteractive'],
    install_requires=[
        'numpy', 'opencv-python', 'scikit-learn', 'scikit-image', 'networkx',
        'scipy'
    ],
    setup_requires=['pytest-runner'],
    test_require=['pytest'],
    include_package_data=True,
    zip_safe=False)
