#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."

python main.py --model qwen2.5:32b --baseline zerocot --benchmark gameof24

python main.py --model qwen2.5:32b --baseline zerocot_single --benchmark mgsm

python main.py --model qwen2.5:32b --baseline zerocot_single --benchmark bigbenchhard --bigbenchhard_task geometric_shapes

python main.py --model qwen2.5:32b --baseline zerocot_single --benchmark bigbenchhard --bigbenchhard_task multistep_arithmetic_two

python main.py --model qwen2.5:32b --baseline zerocot_single --benchmark cruxeval

python main.py --model qwen2.5:32b --baseline zerocot_single --benchmark programmingpuzzles

python main.py --model qwen2.5:32b --baseline zerocot --benchmark bigbenchhard --bigbenchhard_task movie_recommendation

python main.py --model qwen2.5:32b --baseline zerocot --benchmark bigbenchhard --bigbenchhard_task penguins_in_a_table

python main.py --model qwen2.5:32b --baseline zerocot_single --benchmark bigbenchhard --bigbenchhard_task word_sorting

python main.py --model qwen2.5:32b --baseline zerocot --benchmark bigbenchhard --bigbenchhard_task date_understanding