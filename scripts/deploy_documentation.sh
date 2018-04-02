#!/bin/bash

pip install mkdocs pydoc-markdown mkdocs-material
cd docs || exit 1
./build_docs.sh
touch site/.nojekyll
