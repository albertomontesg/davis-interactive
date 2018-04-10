from setuptools import setup

setup(
    install_requires=[
        'numpy>=1.12.1', 'scikit-learn>=0.18', 'scikit-image>=0.13.1',
        'networkx>=2.0', 'scipy>=1.0.0', 'pandas>=0.21.1', 'absl-py>=0.1.13',
        'Pillow>=4.1.1'
    ],
    setup_requires=['pytest-runner'],
    test_require=['pytest', 'pytest-cov'],
)
