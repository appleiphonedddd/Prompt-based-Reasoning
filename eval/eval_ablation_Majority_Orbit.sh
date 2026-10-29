#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."

python main.py --model granite4.1:30b --baseline fot --benchmark bigbenchhard --bigbenchhard_task multistep_arithmetic_two --no_fot_witness_history

python main.py --model granite4.1:30b --baseline fot --benchmark bigbenchhard --bigbenchhard_task geometric_shapes --no_fot_witness_history

python main.py --model granite4.1:30b --baseline fot --benchmark mgsm --no_fot_witness_history

python main.py --model granite4.1:30b --baseline fot --benchmark bigbenchhard --bigbenchhard_task date_understanding --no_fot_witness_history

python main.py --model granite4.1:30b --baseline fot --benchmark bigbenchhard --bigbenchhard_task penguins_in_a_table --no_fot_witness_history

python main.py --model granite4.1:30b --baseline fot --benchmark sonnetwriting --no_fot_witness_history