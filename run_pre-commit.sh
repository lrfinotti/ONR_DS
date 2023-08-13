#! /bin/bash

# MOD_FILES=$(git diff --cached --name-only --diff-filter=M | grep ipynb$)

# check if notebooks were modified
MOD_FILES=$(git ls-files -m *ipynb)


if [[ ! -z $MOD_FILES ]]; then
    echo -e "Notebooks changed!\n"

    for nb in $MOD_FILES; do
        echo "Running ${nb}..."
        python3 -m nbconvert --execute --inplace $nb
        echo -e "Done!\n"
    done

    echo "Zipping notebooks..."
    zip notebooks.zip $MOD_FILES
    echo "Done."
fi
