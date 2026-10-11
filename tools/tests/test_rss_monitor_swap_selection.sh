#!/usr/bin/env bash
# Gate for tools/rss_monitor.sh — the two defects found on 2026-10-09:
#
#   1. The format header was emitted only when the log was "freshly rotated"
#      (size < 200 B), so rss.log carried the 10-column v1 header above
#      11-column rows for its entire 71 MB life.
#   2. Process selection was by ``comm`` (python3) plus an RSS floor, which
#      is blind to a paged-out process: swap is *subtracted* from RSS, so the
#      biggest swap holder on the box (serena, RSS 4.4 MB / VmSwap 554 MB)
#      appeared in neither group.
#
# The script is run FOR REAL. Only its view of the world is faked: a stub
# /proc tree plus stub pgrep/ps on PATH. Nothing here writes to the real
# ~/.deep_worker — HOME is redirected into a temp dir.

set -u

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
MONITOR="${SCRIPT_DIR}/../rss_monitor.sh"

if [ ! -r "$MONITOR" ]; then
    echo "FATAL: cannot read monitor at $MONITOR" >&2
    exit 2
fi

failures=0
checks=0

fail() { echo "  FAIL: $*"; failures=$((failures + 1)); }
ok()   { echo "  ok: $*"; }
check() {
    # check <description> <condition-exit-code-producing-command...>
    local desc=$1; shift
    checks=$((checks + 1))
    if "$@"; then ok "$desc"; else fail "$desc"; fi
}

WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT

# ---------------------------------------------------------------------------
# Fixture: a stub /proc with processes whose shape is taken from the REAL
# measurement on 2026-10-09, not invented.
#
#   pid 2352   serena   rss     4.4 MB  swap 554.2 MB  <- inverted: the §1 case
#   pid 1296179 java    rss  2262.8 MB  swap 189.4 MB  <- non-python3, no swap floor hit
#   pid 330413 python3  rss    86.2 MB  swap 419.1 MB  <- low RSS, known agent
#   pid 1031   python3  rss   361.9 MB  swap 262.3 MB  <- over both floors
#   pid 999001 python3  rss   510.0 MB  swap   0.0 MB  <- NEGATIVE CONTROL: must
#   pid 999002 python3  rss   650.0 MB  swap   0.0 MB     appear (RSS floor) but
#                                                          never as a swap holder
#   pid 999003 python3  rss     3.0 MB  swap   0.0 MB  <- must be FILTERED OUT
#   pid 999004 node     rss     2.0 MB  swap   1.0 MB  <- must be FILTERED OUT
# ---------------------------------------------------------------------------
make_proc() {
    # make_proc <root> <pid> <comm> <rss_kb> <swap_kb> <cmdline>
    local root=$1 pid=$2 comm=$3 rss=$4 swap=$5 cmdline=$6
    local d="${root}/${pid}"
    mkdir -p "$d"
    {
        echo "Name:	${comm}"
        echo "State:	S (sleeping)"
        echo "Threads:	4"
        echo "VmPeak:	$((rss * 2)) kB"
        echo "VmSize:	$((rss * 2)) kB"
        echo "VmRSS:	${rss} kB"
        if [ "$swap" != "-" ]; then
            echo "VmSwap:	${swap} kB"
        fi
    } > "$d/status"
    printf '%s' "$cmdline" | tr ' ' '\0' > "$d/cmdline"
}

build_fixture() {
    local root=$1
    rm -rf "$root"; mkdir -p "$root"
    make_proc "$root" 2352    serena   4505    567500 'serena-mcp-server --transport stdio'
    make_proc "$root" 1296179 java     2317107 193946 'java -jar /opt/lsp/server.jar'
    make_proc "$root" 330413  python3  88269   429158 'python3 -m deep_worker_cli.cli.agent_command rg2_reviewer'
    make_proc "$root" 1031    python3  370586  268595 'python3 -m fix_service.main'
    make_proc "$root" 999001  python3  522240  0      'python3 -m some.heavy.helper'
    make_proc "$root" 999002  python3  665600  0      'python3 -m other.heavy.helper'
    make_proc "$root" 999003  python3  3072    0      'python3 -c pass'
    make_proc "$root" 999004  node     2048    1024   'node /opt/tool/index.js'
}

# Stub pgrep/ps that answer from the fixture, mirroring the real flags the
# script uses: ``pgrep -f <regex>``, ``pgrep -x python3``, ``ps -o etime= -p``.
make_stubs() {
    local bindir=$1 proc_root=$2
    mkdir -p "$bindir"
    cat > "${bindir}/pgrep" <<STUB
#!/usr/bin/env bash
set -u
PROC_ROOT="${proc_root}"
mode=""; pattern=""
while [ \$# -gt 0 ]; do
    case "\$1" in
        -f) mode=full ;;
        -x) mode=exact ;;
        *) pattern=\$1 ;;
    esac
    shift
done
for d in "\$PROC_ROOT"/[0-9]*; do
    [ -d "\$d" ] || continue
    pid=\${d#\$PROC_ROOT/}
    if [ "\$mode" = exact ]; then
        comm=\$(awk '/^Name:/{print \$2}' "\$d/status" 2>/dev/null)
        [ "\$comm" = "\$pattern" ] && echo "\$pid"
    else
        cmdline=\$(tr '\\0' ' ' < "\$d/cmdline" 2>/dev/null)
        echo "\$cmdline" | grep -Eq "\$pattern" && echo "\$pid"
    fi
done
exit 0
STUB
    cat > "${bindir}/ps" <<'STUB'
#!/usr/bin/env bash
echo "   10:00"
exit 0
STUB
    chmod +x "${bindir}/pgrep" "${bindir}/ps"
}

# Run the monitor (or a mutated copy of it) against the fixture and echo the
# path of the log it produced.
run_monitor() {
    # run_monitor <script-path> <run-dir> [preexisting-log-content]
    local script=$1 rundir=$2 seed=${3-}
    local proc_root="${rundir}/proc" bindir="${rundir}/bin" home="${rundir}/home"
    build_fixture "$proc_root"
    make_stubs "$bindir" "$proc_root"
    mkdir -p "${home}/.deep_worker/logs"
    local log="${home}/.deep_worker/logs/rss.log"
    if [ -n "$seed" ]; then printf '%s' "$seed" > "$log"; fi

    # The script hardcodes /proc/PID/status, so redirect that one path by
    # running it with the fixture bind-mounted via a sed'd copy. We rewrite
    # only the procfs ROOT, never the logic under test.
    local staged="${rundir}/monitor_under_test.sh"
    sed -e "s#/proc/#${proc_root}/#g" "$script" > "$staged"
    chmod +x "$staged"

    HOME="$home" PATH="${bindir}:${PATH}" bash "$staged"
    echo "$log"
}

data_rows() { grep -v '^#' "$1" 2>/dev/null | awk 'NF'; }
has_pid()   { data_rows "$1" | awk -v p="$2" '$3 == p { found = 1 } END { exit !found }'; }

echo "=== §2: selection by VmSwap, independent of comm and of the RSS floor ==="
RUN1="${WORK}/run1"; mkdir -p "$RUN1"
LOG1=$(run_monitor "$MONITOR" "$RUN1")

# Positive control part A: the sample is NOT empty. A filter that returns
# nothing is indistinguishable from a filter that is simply broken.
rowcount=$(data_rows "$LOG1" | wc -l)
checks=$((checks + 1))
if [ "$rowcount" -ge 5 ]; then
    ok "non-empty sample: ${rowcount} data rows"
else
    fail "sample too small (${rowcount} rows) — filter may be broken, not merely quiet"
    data_rows "$LOG1" | sed 's/^/      /'
fi

# The defect itself: the inverted-ratio, non-python3 process must be present.
#
# DO NOT "simplify" this to "some row with high swap exists". That weaker
# assertion PASSES ON THE MUTANT: with swap-based selection removed, the
# pre-existing pgrep paths still log seven processes above 200 MB of swap.
# That is exactly why the defect survived unnoticed — the log did not look
# empty, it looked like it covered swap, while omitting the single largest
# holder. Group 3 is load-bearing for one process today, and that process is
# the №1 holder, so the assertion has to name it.
check "serena (inverted ratio, comm != python3) is logged" \
    has_pid "$LOG1" 2352
check "low-RSS known agent rg2_reviewer is logged" \
    has_pid "$LOG1" 330413

# java, as sampled on 2026-10-09, sat just below the swap floor while holding
# multi-GB RSS. It is the sharpest negative control available: a non-python3
# process excluded for the right reason — its swap, not its comm. If selection
# ever reverts to a comm whitelist, or the floor is loosened by accident, this
# is the assertion that notices. The fixture pins the numbers; the live
# process drifts, so no absolute is asserted as a property here.
checks=$((checks + 1))
if has_pid "$LOG1" 1296179; then
    fail "java (sub-floor swap, large RSS) was logged — swap floor not respected"
else
    ok "java excluded on swap (below floor), not on comm"
fi

# Positive control part B: the filter must also be able to stay SILENT.
checks=$((checks + 1))
if has_pid "$LOG1" 999003; then
    fail "tiny no-swap helper (rss 3MB, swap 0) leaked into the log"
else
    ok "tiny no-swap helper correctly filtered out"
fi
checks=$((checks + 1))
if has_pid "$LOG1" 999004; then
    fail "node with 1MB swap leaked in — swap floor is not being applied"
else
    ok "sub-floor swap (node, 1MB) correctly filtered out"
fi

# Known-green inputs: high RSS, zero swap. They must appear (RSS floor) and
# must carry swap=0, i.e. the new criterion did not invent swap for them.
for pid in 999001 999002; do
    check "high-RSS zero-swap helper ${pid} is logged" has_pid "$LOG1" "$pid"
    checks=$((checks + 1))
    swapcol=$(data_rows "$LOG1" | awk -v p="$pid" '$3 == p { print $11; exit }')
    if [ "$swapcol" = "0" ]; then
        ok "helper ${pid} reports swap=0 (not alarmed on)"
    else
        fail "helper ${pid} swap column is '${swapcol}', expected 0"
    fi
done

echo
echo "=== §2 mutation: remove the VmSwap selection -> must go red BY CONTENT ==="
# Mutate BOTH axes of the fix: the swap-based pid discovery AND the RSS-floor
# bypass. A mutant must fail because the row is ABSENT, not because bash threw.
MUT="${WORK}/mutant_no_swap_selection.sh"
# Blank out the swap_pids producer and drop the swap clause from the RSS
# floor, leaving valid bash so the failure is semantic, not syntactic.
python3 - "$MONITOR" "$MUT" <<'PY'
import re, sys
src = open(sys.argv[1]).read()
# axis 1: swap-based discovery yields nothing
src = re.sub(r'swap_pids=\$\(\n.*?\n\)\n', 'swap_pids=""\n', src, flags=re.S)
# axis 2: the RSS floor no longer spares swap holders
src = src.replace(
    '''            if [ "${rss:-0}" -lt "$OTHER_RSS_FLOOR_KB" ] \\
                && [ "${swap:-0}" -lt "$SWAP_FLOOR_KB" ]; then''',
    '''            if [ "${rss:-0}" -lt "$OTHER_RSS_FLOOR_KB" ]; then''')
open(sys.argv[2], 'w').write(src)
PY

checks=$((checks + 1))
if ! bash -n "$MUT" 2>/dev/null; then
    fail "mutant is not valid bash — a syntax error proves nothing"
else
    ok "mutant parses (its failure will be semantic, not syntactic)"
fi

RUN2="${WORK}/run2"; mkdir -p "$RUN2"
LOG2=$(run_monitor "$MUT" "$RUN2")
mut_rows=$(data_rows "$LOG2" | wc -l)

checks=$((checks + 1))
if [ "$mut_rows" -eq 0 ]; then
    fail "mutant produced NO rows at all — cannot distinguish 'defect detected'
        from 'script crashed'; the mutation test is vacuous"
else
    ok "mutant still produces rows (${mut_rows}) — it runs, it just mis-selects"
    checks=$((checks + 1))
    if has_pid "$LOG2" 2352; then
        fail "mutant still logs serena — the gate does not actually detect the defect"
    else
        ok "mutant drops serena: absence is the failure mode, as intended"
    fi
    checks=$((checks + 1))
    if has_pid "$LOG2" 999001; then
        ok "mutant keeps the unrelated high-RSS row (failure is scoped to swap)"
    else
        fail "mutant lost unrelated rows too — mutation is not scoped to the fix"
    fi
fi

echo
echo "=== §1: versioned append-only format header ==="
# A log that already holds the v1 header above v2-shaped rows: exactly the
# 71 MB rss.log on disk. It is far larger than the old 200 B threshold.
OLD_HEADER='# format: ts  kind  pid  rss_kb  vmpeak_kb  vmsize_kb  threads  state  etime  comm'
SEED="${OLD_HEADER}
2026-10-01T00:00:00+02:00 other  12345   1000       2000       2000       4    S  10:00        legacy_row               4242
"
RUN3="${WORK}/run3"; mkdir -p "$RUN3"
LOG3=$(run_monitor "$MONITOR" "$RUN3" "$SEED")

checks=$((checks + 1))
if head -1 "$LOG3" | grep -qx -- "$OLD_HEADER"; then
    ok "append-only honoured: the original v1 header is still the first line"
else
    fail "the pre-existing header was rewritten or dropped — append-only violated"
fi

checks=$((checks + 1))
if grep -q 'legacy_row' "$LOG3"; then
    ok "append-only honoured: pre-existing data row is still readable"
else
    fail "pre-existing data row was lost"
fi

checks=$((checks + 1))
newest=$(grep '^# format' "$LOG3" | tail -1)
if echo "$newest" | grep -q 'swap_kb'; then
    ok "last '# format' line documents swap_kb: ${newest##*format }"
else
    fail "last '# format' line does not mention swap_kb: '${newest}'"
fi

checks=$((checks + 1))
nheaders=$(grep -c '^# format' "$LOG3")
if [ "$nheaders" -eq 2 ]; then
    ok "exactly one new header appended (total ${nheaders})"
else
    fail "expected 2 '# format' lines, found ${nheaders}"
fi

checks=$((checks + 1))
if grep -q '^# NOTE .*layout changed below this line' "$LOG3"; then
    ok "layout change is explicitly marked for readers"
else
    fail "no NOTE marking the layout change"
fi

# The header must agree with the rows beneath it — the actual §1 defect.
checks=$((checks + 1))
hdr_cols=$(grep '^# format' "$LOG3" | tail -1 | sed 's/^# format v[0-9]*: //' | awk '{print NF}')
row_cols=$(data_rows "$LOG3" | tail -1 | awk '{print NF}')
if [ "$hdr_cols" -eq "$row_cols" ]; then
    ok "header column count (${hdr_cols}) == data row field count (${row_cols})"
else
    fail "header claims ${hdr_cols} columns but rows carry ${row_cols} fields"
fi

# Idempotence: a second run on an up-to-date log must NOT append a header.
RUN4="${WORK}/run4"; mkdir -p "$RUN4"
LOG4=$(run_monitor "$MONITOR" "$RUN4")
before=$(grep -c '^# format' "$LOG4")
HOME="${RUN4}/home" PATH="${RUN4}/bin:${PATH}" bash "${RUN4}/monitor_under_test.sh"
after=$(grep -c '^# format' "$LOG4")
checks=$((checks + 1))
if [ "$before" -eq 1 ] && [ "$after" -eq 1 ]; then
    ok "header is not re-emitted when the layout is unchanged (${after} total)"
else
    fail "header count went ${before} -> ${after}; expected to stay at 1"
fi

echo
echo "=== §1 mutation: restore the 'size < 200 B' test -> must go red ==="
MUT2="${WORK}/mutant_size_threshold.sh"
python3 - "$MONITOR" "$MUT2" <<'PY'
import re, sys
src = open(sys.argv[1]).read()
old = re.search(r'last_format=\$\(grep.*?\nfi\n', src, flags=re.S).group(0)
new = ('if [ ! -s "$log_file" ] || '
       '[ "$(stat -c %s "$log_file" 2>/dev/null || echo 0)" -lt 200 ]; then\n'
       '    echo "$FORMAT_LINE"\n'
       'fi\n')
open(sys.argv[2], 'w').write(src.replace(old, new))
PY

checks=$((checks + 1))
if bash -n "$MUT2" 2>/dev/null; then
    ok "size-threshold mutant parses"
else
    fail "size-threshold mutant is not valid bash"
fi

RUN5="${WORK}/run5"; mkdir -p "$RUN5"
LOG5=$(run_monitor "$MUT2" "$RUN5" "$SEED")
checks=$((checks + 1))
mut_rows5=$(data_rows "$LOG5" | wc -l)
if [ "$mut_rows5" -eq 0 ]; then
    fail "size-threshold mutant produced no rows — vacuous mutation"
else
    newest5=$(grep '^# format' "$LOG5" | tail -1)
    if echo "$newest5" | grep -q 'swap_kb'; then
        fail "mutant still documents swap_kb — gate does not detect the defect"
    else
        ok "mutant leaves the stale v1 header as the last one: '${newest5##*format }'"
    fi
fi

echo
echo "=== summary ==="
echo "checks: ${checks}   failures: ${failures}"
[ "$failures" -eq 0 ] || exit 1
echo "PASS"
