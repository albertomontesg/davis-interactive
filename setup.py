from setuptools import setup

setup(
    python_requires='>=2.7, !=3.0.*, !=3.1.*, !=3.2.*, !=3.3.*, <4',
    install_requires=[
        'numpy>=1.12.1', 'scikit-learn>=0.18', 'scikit-image>=0.13.1',
        'networkx>=2.0', 'scipy>=1.0.0', 'pandas>=0.21.1', 'absl-py>=0.1.13',
        'Pillow>=4.1.1', 'mock;python_version<"3.0"',
        'pathlib2;python_version<"3.0"', 'six>=1.10.0'
    ])
