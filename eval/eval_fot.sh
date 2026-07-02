#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."

python main.py --model qwen2.5:32b --baseline fot --benchmark gameof24

python main.py --model qwen2.5:32b --baseline fot --benchmark cruxeval

python main.py --model qwen2.5:32b --baseline fot --benchmark programmingpuzzles

python main.py --model qwen2.5:32b --baseline fot --benchmark bigbenchhard --bigbenchhard_task geometric_shapes

python main.py --model qwen2.5:32b --baseline fot --benchmark mgsm

python main.py --model qwen2.5:32b --baseline fot --benchmark bigbenchhard --bigbenchhard_task multistep_arithmetic_two
