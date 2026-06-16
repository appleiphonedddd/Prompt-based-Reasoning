#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."

python main.py --model qwen2.5:32b --baseline standard --benchmark gameof24

python main.py --model qwen2.5:32b --baseline standard --benchmark mgsm

python main.py --model qwen2.5:32b --baseline standard --benchmark bigbenchhard --bigbenchhard_task geometric_shapes

python main.py --model qwen2.5:32b --baseline standard --benchmark bigbenchhard --bigbenchhard_task multistep_arithmetic_two

python main.py --model qwen2.5:32b --baseline standard --benchmark cruxeval

python main.py --model qwen2.5:32b --baseline standard --benchmark programmingpuzzles