# Build Documentation

## Setup

First install all necessary packages to build the documentation:

```bash
# Python 2
pip install mkdocs pydoc-markdown mkdocs-material
# Python 3
pip3 install mkdocs pydoc-markdown mkdocs-material
```

## Build Doc

Build the documentation locally running the following script:

```bash
./build_docs.sh
```

## Serve Locally

You can serve the documentation locally running:

```bash
pydocmd serve
```

Check the documentation at `http://127.0.0.1:8000`.