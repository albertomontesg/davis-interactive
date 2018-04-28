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

In addition to installing the framework, you will need to download the `trainval` DAVIS 2017 Dataset with 480p resolution from [here](http://davischallenge.org/davis2017/code.html).

Moreover, you can download the `trainval` scribbles from [here](https://data.vision.ee.ethz.ch/csergi/share/DAVIS-Interactive/DAVIS-2017-scribbles-trainval.zip).
You have to unzip the scibbles zip file in the folder containing DAVIS (in /path/to supposing DAVIS is in /path/to/DAVIS).

*Note*: A script to download automatically the dataset and the scribbles is coming soon.

## Install from Source

If you prefer, you can install the package downloading the source code and installing it. To do so, you need to run these commands:

```bash
# Download the code
git clone https://github.com/albertomontesg/davis-interactive.git && cd davis-interactive
# Install it - Python 3 recommended
python setup.py install
```

## Development

If you want to contribute to this package you will need to have a copy of the code to work with. First, download the code from Github:

```bash
git clone https://github.com/albertomontesg/davis-interactive.git && cd davis-interactive
git checkout -b my/new/branch
```

To have a development copy of the package installed for Python you can run the following:

```bash
pip3 install -e .
```

This will link the available copy of the package to your current copy so all the modifications that you made on the code will be visible by any script.

If you want to help us improve the documentation it will be necessary to have some additional packages:

```bash
pip3 install mkdocs pydoc-markdown mkdocs-material
```

Then you serve the documentation live on your local machine to check the changes you make on the documentation.

```bash
cd docs

# Serve the documentation live
pydocmd serve

# Build the documentation
./build_docs.sh
```

