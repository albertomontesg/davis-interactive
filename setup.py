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
        'numpy>=1.12.1', 'scikit-learn>=0.18', 'scikit-image>=0.13.1',
        'networkx>=2.0', 'scipy>=1.0.0', 'pandas>=0.21.1'
    ],
    setup_requires=['pytest-runner'],
    test_require=['pytest', 'pytest-cov'],
    include_package_data=True,
    zip_safe=False)
