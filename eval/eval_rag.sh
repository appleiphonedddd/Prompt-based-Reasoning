#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."

# Table 4.6 — knowledge-intensive QA: retrieval strategy x reasoning (CoT / FoT).
#
# Every row of one table must use the same LLM *and backend*, so that differences
# reflect the retrieval strategy alone. DRAGIN reads token entropies and attention,
# which only a local Hugging Face model exposes, so the full table (all four
# retrieval rows) needs BACKEND=hf:
#
#   BACKEND=hf MODEL=Qwen/Qwen2.5-7B-Instruct ./eval/eval_rag.sh
#
# With the default Ollama backend the DRAGIN rows are skipped.
#
# The first run builds the BM25 index over the DPR Wikipedia dump (21M passages,
# ~1 h, one-time). For a quick pass use CORPUS=dataset (IRCoT's per-dataset
# corpora) and fewer questions, e.g. NUM_SAMPLES=100.

BACKEND=${BACKEND:-ollama}
MODEL=${MODEL:-qwen2.5:32b}
CORPUS=${CORPUS:-wiki}
NUM_SAMPLES=${NUM_SAMPLES:-1000}
WORKERS=${WORKERS:-1}

RETRIEVALS="none single ircot"
if [ "$BACKEND" = "hf" ]; then
  RETRIEVALS="$RETRIEVALS dragin"
fi

for dataset in hotpotqa musique 2wikimqa iirc strategyqa; do
  for retrieval in $RETRIEVALS; do
    for reasoning in cot fot; do
      python RAG/eval.py --backend "$BACKEND" --model "$MODEL" --corpus "$CORPUS" \
        --num_samples "$NUM_SAMPLES" --workers "$WORKERS" \
        --dataset "$dataset" --retrieval "$retrieval" --reasoning "$reasoning"
    done
  done
done

# Ablations — uncomment to run.
#
# (a) IRCoT reading its answer off the retrieval CoT instead of a separate reader
#     (IRCoT App. F, Table 6).
# python RAG/eval.py --dataset hotpotqa --retrieval ircot --ircot_reader none
#
# (b) FoT's pilot trigger: a single violation repairs, no orbit-majority rule.
# python RAG/eval.py --dataset hotpotqa --retrieval single --reasoning fot \
#   --fot_tau 1 --no_fot_orbit_majority
#
# (c) FoT with only the programmatic / only the reliability-increasing relation.
# python RAG/eval.py --dataset 2wikimqa --retrieval single --reasoning fot \
#   --fot_relations qa_swap_operands
# python RAG/eval.py --dataset hotpotqa --retrieval single --reasoning fot \
#   --fot_relations qa_decompose --fot_tau 1
#
# (d) DRAGIN's RIND threshold sweep (DRAGIN Table 5).
# for t in 0.6 0.8 1.0 1.2 1.4; do
#   python RAG/eval.py --backend hf --model Qwen/Qwen2.5-7B-Instruct \
#     --dataset hotpotqa --retrieval dragin --dragin_threshold $t
# done
