#!/bin/bash

cd docs || exit 1
sed -i -e s/\$GOOGLE_ANALYTICS/$GOOGLE_ANALYTICS/ mkdocs.yml
./build_docs.sh
touch site/.nojekyll
echo "$DOCUMENTATION_DOMAIN" >>site/CNAME
