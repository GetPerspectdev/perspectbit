#!/usr/bin/env bash
set -euo pipefail
# This will format files on commit.
# The commit will fail, but the files will be updated.
# Just commit again and it should work, assuming the issues were only formatting.
#
#
# Best used by adding as a pre-commit hook.
# See setupHooks.sh

formatEndpoint=$(git config --get modelbit.restendpoint || exit 0)
formatEndpoint="${formatEndpoint:-https://app.modelbit.com/api/format}"

fixFiles=true
showDiff=false
silent=false

usage() {
    echo "Usage: $0 [-c] [-d] [-s]"
    echo "-c: Check and not rewrite files"
    echo "-d: Display diff of changes instead of just a message"
    echo "-s: Silent, no output"
    exit 1
}

parseArgs() {
    while getopts cdsh flag; do
        case "${flag}" in
            c) fixFiles=false ;;
            d) showDiff=true ;;
            s) silent=true ;;
            *) usage ;;
        esac
    done
}

setupTmp() {
    tmp_dir=$(mktemp -d -t ci-XXXXXXXXXX)
    if [ -z "$tmp_dir" ]; then
        echo >&2 "Failed to create tmpdir"
        exit 1
    else
        trap 'rm -rf -- "$tmp_dir"' EXIT
    fi
}

print() {
    [[ $silent != "true" ]] && echo >&2 "$1"
}

# checkFile returns 1 if the file doesn't match the version returned from
# the format endpoint. If -d is passed, print a diff of the files.
checkFile() {
    local fname=$1
    local failed=0
    if ! cmp --silent "$tmp_dir/$fname" "$fname"; then
        failed=1
        print "File $fname is not valid"
        if [[ $showDiff == "true" ]]; then
            diff --color -U 3 "$fname" "$tmp_dir/$fname" >&2
        fi
    fi
    if [[ $fixFiles == "true" ]]; then
        cp "$tmp_dir/$fname" "$fname"
    fi

    return $failed
}

# Expected response format:
# Header row is either FILE: or ERROR:
# File contents follows FILE: until we see string EOF
# e.g.
# FILE: path/to/file.yaml
# fileContents
# over
# many lines
# EOF
# FILE: path/tofile.yaml
# content
# EOF
# ERROR: path/to/file.yaml Error Message here

exitWithError() {
    echo >&2 "Failed API request to ${formatEndpoint}"
    exit 1
}

findFiles() {
    find . -name \*.yaml | awk -F "\n" '{print "-F \042"$1"\042=@\042"$1"\042"}' | xargs curl -fs "${formatEndpoint}" || exitWithError
}

formatFiles() {
    local curFile="/dev/null"
    findFiles | (
        local failed=0
        # Parse response line-by-line
        while IFS= read -r line; do
            # New file header
            if [[ "$line" =~ FILE:\ (.+) ]]; then
                curFile="${BASH_REMATCH[1]}"
                mkdir -p "$(dirname "$tmp_dir/$curFile")"
            # End of file
            elif [[ "$line" == "EOF" ]]; then
                if [[ $curFile == "/dev/null" ]]; then
                    # Shouldn't happen? Only if we get file data before a valid header.
                    print "Bad response, Got data before first file header"
                    return 1
                fi
                if ! checkFile "$curFile"; then
                    failed=1
                fi
            # Error header
            elif [[ "$line" =~ ERROR:\ (.+)\|(.+) ]]; then
                failed=1
                local filename=${BASH_REMATCH[1]}
                local error=${BASH_REMATCH[2]}
                print "Error in ${filename}: ${error}"
            # File content
            else
                echo "$line" >>"$tmp_dir/$curFile"
            fi
        done
        return $failed
    )
}

main() {
    parseArgs "$@"
    setupTmp
    return $(formatFiles)
}

main "$@"
