#!/usr/bin/env bash
# Build dataset v2 (docs/dataset-v2-plan.md): teacher candidates for the in-scope cite twins, the
# same filter as v1, then v2 = v1 (unchanged) + the kept twins, checked and committed.
#
#   scripts/build_v2.sh [--filter opus|gpt-5.5]             dry run (the default): no request,
#                                                           nothing written
#   scripts/build_v2.sh --approved --filter opus|gpt-5.5    the paid run, then a commit
#
# The teacher and filter-judge calls are paid, so they run only with --approved, given after the
# owner approves the spend. --filter picks the filter judge (v1 was filtered by Opus). distill.py
# has no spend cap, so the approved run does the dry run first and stops when the planned teacher
# calls exceed MAX_TEACHER_CALLS. An interrupted run is rerun with the same command: distill.py
# resumes the missing candidates and already-filtered additions are reused.
set -euo pipefail

cd "$(dirname "$0")/.."

SCENARIOS=data/scenarios/train_v2_additions.jsonl
MAX_TEACHER_CALLS=210
DISTILL_OUT=data/distill/v2_additions
CANDIDATES=$DISTILL_OUT/candidates.jsonl
V1=data/datasets/v1
V1_SHA256=ae470ee1a6eb771bae3e7f1b4dcaa5764e457e101b4023ad101711d7afcbd179
V1_COUNT=448
V2=data/datasets/v2
ADDITIONS=$V2/additions
EVAL_SETS=(
    data/scenarios/ablation.jsonl data/scenarios/eval_own.jsonl data/scenarios/heldout.jsonl
    data/scenarios/rollout_pilot.jsonl
)
FEW_SHOT=data/few_shot/examples.jsonl

usage() {
    echo "usage: scripts/build_v2.sh [--approved] [--filter opus|gpt-5.5]" >&2
    exit 2
}

approved=0
filter=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --approved) approved=1; shift ;;
        --filter) [[ $# -ge 2 ]] || usage; filter="$2"; shift 2 ;;
        *) echo "error: unknown argument: $1" >&2; usage ;;
    esac
done

# The two filter judges with pinned ids (config/models.toml).
case "$filter" in
    opus) judge=anthropic:claude-opus-5-5 ;;
    gpt-5.5) judge=openai:gpt-5.5-2026-04-23 ;;
    "") judge="" ;;
    *) echo "error: --filter must be opus or gpt-5.5, not '$filter'" >&2; exit 2 ;;
esac
if (( approved )) && [[ -z $judge ]]; then
    echo "error: --approved needs --filter opus or --filter gpt-5.5" >&2
    exit 2
fi

step() { echo; echo "==> $*"; }
run() { echo "+ $*"; "$@"; }

if (( approved )); then
    step "preflight (no paid call yet)"
    branch=$(git symbolic-ref --quiet --short HEAD || true)
    if [[ -z $branch || $branch == main ]]; then
        echo "error: run on a feature branch, not '${branch:-detached HEAD}'" >&2
        exit 1
    fi
    # Only this script's own outputs may be uncommitted (a resumed run).
    if [[ -n $(git status --porcelain -- . ":!$DISTILL_OUT" ":!$V2") ]]; then
        echo "error: the worktree has uncommitted changes; commit or stash them first" >&2
        exit 1
    fi
    for name in train.jsonl manifest.json; do
        if [[ -e $V2/$name ]]; then
            echo "error: $V2/$name already exists; datasets are never overwritten" >&2
            exit 1
        fi
    done
    # Fails here, before any spend, when the scrub terms are missing (SCRUB_TERMS_PATH).
    run uv run python scripts/scrub_check.py --repo .
fi

resume=()
if [[ -s $CANDIDATES ]]; then
    resume=(--resume)
fi
distill=(
    uv run python scripts/distill.py --scenarios "$SCENARIOS"
    --teacher anthropic:claude-opus-5-5 --k 2 --out "$DISTILL_OUT" ${resume[@]+"${resume[@]}"}
)

step "teacher plan (distill.py --dry-run: no request, nothing written)"
echo "+ ${distill[*]} --dry-run"
plan=$("${distill[@]}" --dry-run)
echo "$plan"
calls=$(sed -n 's/^teacher calls: //p' <<<"$plan")
if [[ ! $calls =~ ^[0-9]+$ ]]; then
    echo "error: distill.py --dry-run printed no teacher call count" >&2
    exit 1
fi
if (( calls > MAX_TEACHER_CALLS )); then
    echo "error: $calls teacher calls planned, more than the approved $MAX_TEACHER_CALLS" >&2
    exit 1
fi

step "filter plan"
echo "filter judge: ${judge:-(set with --filter opus|gpt-5.5)}"
echo "judge calls: a hint screen and a judge call per candidate that passes the deterministic"
echo "             checks (at most $((2 * MAX_TEACHER_CALLS)))"
echo "then: $ADDITIONS (build_dataset.py),"
echo "      $V2 = $V1 + kept twins (assemble_dataset.py),"
echo "      diff_datasets.py $V1 $V2, leakage, secret scan, scrub check, git commit"

if (( ! approved )); then
    echo
    echo "dry run: no requests made, nothing written."
    echo "After the owner approves the spend: scripts/build_v2.sh --approved --filter opus|gpt-5.5"
    exit 0
fi

step "teacher candidates ($calls calls)"
if (( calls > 0 )); then
    run "${distill[@]}"
fi
uv run python - "$CANDIDATES" "$(($(grep -c . "$SCENARIOS") * 2))" <<'EOF'
import json
import sys

path, expected = sys.argv[1], int(sys.argv[2])
rows = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
errors = sum(r["error"] is not None for r in rows)
print(f"{len(rows)} candidates, {errors} teacher errors")
if len(rows) != expected:
    sys.exit(f"error: expected {expected} candidates in {path}")
EOF

step "filter ($judge)"
if [[ -f $ADDITIONS/manifest.json ]]; then
    echo "reusing the filtered additions in $ADDITIONS"
else
    run uv run python scripts/build_dataset.py --candidates "$CANDIDATES" --out "$ADDITIONS" \
        --judge "$judge"
fi

step "assemble $V2"
run uv run python scripts/assemble_dataset.py --base "$V1" --additions "$ADDITIONS" --out "$V2" \
    --expect-base-sha256 "$V1_SHA256"

step "diff $V1 -> $V2"
run uv run python scripts/diff_datasets.py "$V1" "$V2" --out "$V2/diff_v1.md" \
    --json "$V2/diff_v1.json"
uv run python - "$V2/diff_v1.json" "$V1_COUNT" <<'EOF'
import json
import sys

summary = json.load(open(sys.argv[1], encoding="utf-8"))
if summary["removed"] or summary["relabeled"] or summary["unchanged"] != int(sys.argv[2]):
    sys.exit("error: v2 must only add examples: every v1 example unchanged")
print(f"only added: {len(summary['added'])} examples; {summary['unchanged']} v1 examples unchanged")
EOF

step "leakage"
run uv run python scripts/check_leakage.py --train "$V2/train.jsonl" --eval "${EVAL_SETS[@]}" \
    --report "$V2/leakage_report.md"
run uv run python scripts/check_leakage.py --train "$SCENARIOS" --eval "${EVAL_SETS[@]}" \
    --few-shot "$FEW_SHOT"

step "secret scan and scrub check"
run git add "$DISTILL_OUT" "$V2"
run uv run python scripts/secret_scan.py --repo .
run uv run python scripts/scrub_check.py --repo .

step "commit"
kept=$(uv run python -c "import json; print(json.load(open('$ADDITIONS/manifest.json'))['kept'])")
run git commit --quiet -m "data: build dataset v2 (v1 + $kept in-scope cite twins, $filter filter)"
git log --oneline -1
echo "dataset v2 built and committed. Next: scripts/check.sh, then the release and re-pin."
