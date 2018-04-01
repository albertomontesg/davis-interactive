#!/bin/bash

[[ -d _build/docs/images ]] && rm -r _build/docs/images

cp -r images _build/docs/images
pydocmd build
