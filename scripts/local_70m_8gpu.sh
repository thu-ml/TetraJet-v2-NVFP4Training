#!/usr/bin/env bash
set -eo pipefail

source "$(dirname "$0")/common.sh"
run_single_node 70m_50bt "$1" "$2"
