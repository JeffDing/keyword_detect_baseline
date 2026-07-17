#!/bin/bash
cd "$(dirname "$0")"
source /usr/local/Ascend/cann-8.5.1/set_env.sh
export ASCEND_VISIBLE_DEVICES=0
exec python3 infer.py "$@"
