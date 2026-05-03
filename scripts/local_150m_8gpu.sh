#!/usr/bin/env bash
set -eo pipefail

source "$(dirname "$0")/common.sh"
run_single_node 150m_100bt "$1" "$2"
