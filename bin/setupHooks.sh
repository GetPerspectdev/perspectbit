#!/usr/bin/env bash

cd .git/hooks || exit 1
ln -s ../../bin/format.sh pre-commit
