#!/bin/bash

python setup.py clean --all
python setup sdist bdist_wheel

# Upload to PyPi test
twine upload --repository-url "https://test.pypi.org/legacy/" \
	-u $PYPI_USER -p $PYPI_PASSWORD dist/*
# Upload to PyPi
twine upload -r pypi -u $PYPI_USER -p $PYPI_PASSWORD dist/*
