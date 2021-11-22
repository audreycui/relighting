#!/bin/bash

# Start from directory of script
cd "$(dirname "${BASH_SOURCE[0]}")"

# Set up git config filters so huge output of notebooks is not committed.
git config filter.clean_ipynb.clean "$(pwd)/ipynb_drop_output.py"
git config filter.clean_ipynb.smudge cat
git config filter.clean_ipynb.required true

# Set up symlinks for the example notebooks
# mkdir -p ../datasets
# mkdir -p ../results
# ln -sfn ../datasets -t .
# ln -sfn ../results -t .
# ln -sfn ../models -t .
# ln -sfn ../utils -t .
# ln -sfn ../torch_utils -t .
# ln -sfn ../dnnlib -t .
# ln -sfn ../experiment -t .
# ln -sfn ../evaldata -t .
