# Installation

To use this package it is necessary to install it (with `pip` or by source) and also download the DAVIS 2017 Dataset. The instructions to do so are as follows.

## PyPi Install

To install the package you can run the following command on your terminal:

```bash
# Python 3 strongly recommended
# Install numpy and Cython as are required to build our package
pip install numpy Cython
# Install the package
pip install davisinteractive
```

## DAVIS Dataset

In addition to installing the framework, you will need to download the `train`, `val` and `test-dev` (for the challenge) DAVIS 2017 subsets with 480p resolution from <a href="http://davischallenge.org/davis2017/code.html" target="_blank">here</a>.

Moreover, you can download the `train` and `val` scribbles from <a href="https://data.vision.ee.ethz.ch/csergi/share/DAVIS-Interactive/DAVIS-2017-scribbles-trainval.zip" target="_blank">here</a>.
You have to unzip the scibbles zip file in the folder containing DAVIS (in /path/to supposing DAVIS is in /path/to/DAVIS).

The scribbles for the `test-dev` are provided directly by the server that is online during the challange periods.

## Install from Source

If you prefer, you can install the package downloading the source code and installing it. To do so, you need to run these commands:

```bash
# Download the code
git clone https://github.com/albertomontesg/davis-interactive.git && cd davis-interactive
# Install it - Python 3 recommended
python setup.py install
```

## Development

If you want to contribute to this package you need to have a copy of the code to work with. First, download the code from Github:

```bash
git clone https://github.com/albertomontesg/davis-interactive.git && cd davis-interactive
git checkout -b my/new/branch
```

To have a development copy of the package installed for Python you can run the following:

```bash
# Python 3 strongly recommended
# Install numpy and Cython as are required to build our package
pip install numpy Cython
pip install -e .
```

This links the available copy of the package to your current copy so all the modifications that you made on the code is visible by any script.

If you want to help us improve the documentation it is necessary to have some additional packages:

```bash
pip install mkdocs pydoc-markdown mkdocs-material requests
```

Then you serve the documentation live in your local machine in order to check the changes that you make in the documentation.

```bash
cd docs

# Serve the documentation live
pydocmd serve

# Build the documentation
./build_docs.sh
```

