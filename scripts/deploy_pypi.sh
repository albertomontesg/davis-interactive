#!/bin/bash

python setup.py clean --all
python setup.py sdist

# Upload to PyPi test
twine upload --repository-url "https://test.pypi.org/legacy/" dist/*
# Upload to PyPi
twine upload dist/*
