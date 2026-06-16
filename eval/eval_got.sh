#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."

python main.py --model qwen2.5:32b --baseline got --benchmark gameof24 --got_branches 3 --got_keep 1 --got_refine 0

python main.py --model qwen2.5:32b --baseline got --benchmark mgsm --got_branches 3 --got_keep 1 --got_refine 0

python main.py --model qwen2.5:32b --baseline got --benchmark bigbenchhard --bigbenchhard_task geometric_shapes --got_branches 3 --got_keep 1 --got_refine 0

python main.py --model qwen2.5:32b --baseline got --benchmark bigbenchhard --bigbenchhard_task multistep_arithmetic_two --got_branches 3 --got_keep 1 --got_refine 0

python main.py --model qwen2.5:32b --baseline got --benchmark cruxeval --got_branches 3 --got_keep 1 --got_refine 0

python main.py --model qwen2.5:32b --baseline got --benchmark programmingpuzzles --got_branches 3 --got_keep 1 --got_refine 0