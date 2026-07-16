#!/bin/bash
cd "$(dirname "$0")"
source /usr/local/Ascend/cann-8.5.1/set_env.sh
exec python3 train.py "$@"
