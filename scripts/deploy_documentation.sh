#!/bin/bash

pip install mkdocs pydoc-markdown mkdocs-material
cd docs || exit
./build_docs.sh
pydocmd gh-deploy
