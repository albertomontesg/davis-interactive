#!/bin/bash

set -e

[[ -d _build/docs/images ]] && rm -r _build/docs/images

mkdir -p _build/docs
cp -r images _build/docs/images
pydocmd build
