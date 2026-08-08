#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."

# Executable regime (a trusted checker c_q decides; the model only proposes a probe).
# Cost here is O(K*2): one probe settles a deterministic checker's verdict.
python main.py --model qwen2.5:32b --baseline fot --benchmark gameof24

python main.py --model qwen2.5:32b --baseline fot --benchmark cruxeval

python main.py --model qwen2.5:32b --baseline fot --benchmark programmingpuzzles

python main.py --model qwen2.5:32b --baseline fot --benchmark bigbenchhard --bigbenchhard_task multistep_arithmetic_two

# Relational regime (no checker: the candidate is attacked with the relation
# library R_q, and every verdict comes from a deterministic comparator).
# Cost is O(K*(1+m*s)) — with the paper's K=3, m=3, s=5 that is up to 48 short
# calls per question. Lower --fot_survival / --fot_votes to trade rigour for time.
python main.py --model qwen2.5:32b --baseline fot --benchmark bigbenchhard --bigbenchhard_task geometric_shapes

python main.py --model qwen2.5:32b --baseline fot --benchmark mgsm

# Ablations from the paper — uncomment to run.
#
# (a) No voting: a single sample decides each relational check, so one bad
#     completion can refute a correct answer (the failure mode §1 diagnoses).
# python main.py --model qwen2.5:32b --baseline fot --benchmark mgsm --fot_votes 1
#
# (b) Backward substitution only / metamorphic transforms only.
# python main.py --model qwen2.5:32b --baseline fot --benchmark mgsm --fot_relations mask_quantity
# python main.py --model qwen2.5:32b --baseline fot --benchmark mgsm \
#   --fot_relations permute_premises scale_quantities_x2
#
# (c) Model-proposed library (pi_mr-gen) instead of the hand-audited one.
# python main.py --model qwen2.5:32b --baseline fot --benchmark mgsm --fot_generate_relations
#
# (d) Best-corroborated candidate on budget exhaustion instead of Alg. 4's last one.
# python main.py --model qwen2.5:32b --baseline fot --benchmark mgsm --fot_archive
