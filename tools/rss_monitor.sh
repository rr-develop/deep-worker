#!/usr/bin/env bash
# RSS / memory snapshot for the web UI service and every running agent.
#
# Designed to be cheap and side-effect-free:
#   * only reads /proc/PID/status — no ptrace, no py-spy, no inject.
#   * one line per process, append-only.
#   * dies on its own; no daemonization.
#
# The companion systemd user timer (~/.config/systemd/user/dwagents-rss.timer)
# fires this every 5 minutes. Output goes to ~/.deep_worker/logs/rss.log,
# which logrotate trims weekly.
#
# Read the log:
#     tail -F ~/.deep_worker/logs/rss.log
#
# Look at growth of a single PID:
#     awk -v pid=587042 '$3 == pid' ~/.deep_worker/logs/rss.log | head -20
#
# Attribute swap consumption (field 11, kB — appended after comm):
#     awk '$1 ~ /^2026-09-28T17:44/ {printf "%8.0f MB  %s\n", $11/1024, $10}' \
#         ~/.deep_worker/logs/rss.log | sort -nr | head
#
# Why VmSwap is recorded: RSS *excludes* swapped-out pages, and VmPeak is a
# high-water mark that cannot distinguish "paged out" from "freed". During the
# 2026-09-28 outage (swap pinned at 100% for 9 days, load 210, box
# unreachable) the per-process swap holder could NOT be identified from this
# log at all — only VmSwap answers "who is holding the swap right now".
# Rows are emitted with swap=? when the kernel omits VmSwap (kthreads).
#
# Swap is also a SELECTION criterion, not just a recorded column: any process
# holding more than SWAP_FLOOR_KB is logged regardless of its comm and
# regardless of the RSS floor. RSS and swap are inverted for a paged-out
# process (measured 2026-10-09: serena at RSS 4.4 MB / VmSwap 554 MB was the
# single largest swap holder on the box and was invisible to both filters).
#
# The format header is append-only and versioned: when the layout changes, a
# new "# format vN:" line is appended at the point of change rather than the
# file being rewritten. Take the LAST one to learn the current layout.
#
# Why no Python-level monitoring: every leak found so far (six of
# them, see the related internal incident analyses) was visible from
# procfs alone. Adding pyrasite / gc.get_objects to a monitoring loop
# on a multi-runtime (tokio + torch + jemalloc) process is a known
# cause of segfaults and is exactly what cost us several agent
# processes during diagnostics.

set -u

# Ensure the log dir exists BEFORE we redirect stdout/stderr into it —
# otherwise the ``exec >> ...`` below fails silently on a fresh $HOME
# (this bit the test rig during a fix verification).
log_file="${HOME}/.deep_worker/logs/rss.log"
err_file="${HOME}/.deep_worker/logs/rss.err"
mkdir -p "$(dirname "$log_file")"

exec >>"$log_file" 2>>"$err_file"

ts=$(date -Iseconds)

# The current column layout, as a single source of truth. Bump the version
# whenever a column is appended; the check below then re-emits the header.
FORMAT_VERSION=2
FORMAT_LINE="# format v${FORMAT_VERSION}: ts  kind  pid  rss_kb  vmpeak_kb  vmsize_kb  threads  state  etime  comm  swap_kb"

# Emit the header when the log does not already END with this exact layout.
#
# The old test was "file size < 200 B", i.e. "freshly rotated". That silently
# fails the only case that matters: a log that already has MILLIONS of lines
# when a column is appended. rss.log carried the v1 header (10 columns, no
# swap_kb) for its whole life while rows below it had 11 fields from b6c5524
# onward — the artifact's own documentation contradicted the artifact.
#
# Appending rather than rewriting is deliberate: this log is append-only by
# design, and the 2026-09-28 swap incident is reconstructed from its history.
# Rotating or rewriting would destroy that. A reader parsing by NF is
# unaffected; a reader looking for the layout takes the LAST ``# format``
# line, which is the one that applies to the rows beneath it.
last_format=$(grep '^# format' "$log_file" 2>/dev/null | tail -1 || true)
if [ "$last_format" != "$FORMAT_LINE" ]; then
    if [ -s "$log_file" ] && [ -n "$last_format" ]; then
        echo "# NOTE ${ts}: column layout changed below this line; previous layout was: ${last_format}"
    fi
    echo "$FORMAT_LINE"
fi

# RSS floor below which an unmatched python process is uninteresting.
# 200 MB — chosen because every real leak this monitor has ever caught
# was well above that watermark, whereas hundreds of tiny helper /
# fixture children (e.g. leaked pytest lsp stubs ~2–4 MB each) would
# otherwise flood the log.
# See a related internal incident analysis (not included here), which
# recommended a catch-all with attribution: sample every python3/java
# process above an RSS floor (say 200 MB), tagging unrecognised ones
# kind=other.
OTHER_RSS_FLOOR_KB=204800

# Swap floor above which a process is logged NO MATTER WHAT it is.
# 200 MB — measured, not guessed: on 2026-10-09 (3.55 GB of swap in use)
# this caught exactly the 8 real holders and nothing else.
#
# This exists because RSS and swap are INVERTED for a paged-out process:
# serena sat at RSS 4.4 MB / VmSwap 554 MB — a 1:126 ratio — making the
# single largest swap holder on the box invisible to BOTH existing filters
# (not ``python3`` by comm, and far below OTHER_RSS_FLOOR_KB by RSS).
SWAP_FLOOR_KB=204800

# Pick up three groups:
#   1. Known agents / dwweb — regardless of RSS (steady-state baseline).
#   2. Every other python3 process above the RSS floor — the catch-all.
#   3. Every process of ANY kind holding more than SWAP_FLOOR_KB of swap.
# Without the catch-all, detached grandchildren (post-commit reindex,
# repo_graph pipeline helpers with cmdline "python3 -", …) never showed
# up in rss.log even at multi-GB RSS.  A related internal incident
# analysis (not included here) describes the concrete leak this closes.
#
# Group 3 scans procfs directly instead of widening the ``comm`` list to
# ``serena|java|node|…``: that enumeration is a whitelist, and it would
# diverge again at the next tool added to the box (a go binary, a
# rust-analyzer). The property we care about is "holds swap", so select on
# it. Reading VmSwap for every pid is one awk per /proc/PID/status, the
# same cost this script already pays for the sampled set.
known_pids=$(pgrep -f 'python3.*run\.py|python3.*deep_worker_cli\.cli\.agent_command' 2>/dev/null || true)
other_pids=$(pgrep -x 'python3' 2>/dev/null || true)
swap_pids=$(
    for status_file in /proc/[0-9]*/status; do
        swap_kb=$(awk '/^VmSwap:/ { print $2; exit }' "$status_file" 2>/dev/null) || continue
        [ -n "$swap_kb" ] || continue
        if [ "$swap_kb" -ge "$SWAP_FLOOR_KB" ] 2>/dev/null; then
            status_dir=${status_file%/status}
            echo "${status_dir#/proc/}"
        fi
    done
)
all_pids=$(printf '%s\n%s\n%s\n' "$known_pids" "$other_pids" "$swap_pids" \
    | awk 'NF && !seen[$0]++')

for pid in $all_pids; do
    [ -r "/proc/${pid}/status" ] || continue
    # Grab the fields we care about from /proc/PID/status in one read.
    eval "$(awk '
        /^Name:/      { printf "comm=%q\n",   $2 }
        /^State:/     { printf "state=%s\n",  $2 }
        /^Threads:/   { printf "thr=%s\n",    $2 }
        /^VmRSS:/     { printf "rss=%s\n",    $2 }
        /^VmPeak:/    { printf "peak=%s\n",   $2 }
        /^VmSize:/    { printf "vmsz=%s\n",   $2 }
        /^VmSwap:/    { printf "swap=%s\n",   $2 }
    ' "/proc/${pid}/status" 2>/dev/null)"
    etime=$(ps -o etime= -p "$pid" 2>/dev/null | tr -d ' ')
    [ -z "${rss:-}" ] && continue
    # Identify the role of this process from its cmdline.
    # The redirect is wrapped rather than just silencing tr: when a process
    # exits between the pid listing and this read, it is the REDIRECT that
    # fails, and bash reports that on stderr before tr ever runs. Exiting
    # mid-sample is normal, so it must not land in rss.err.
    cmdline=$({ tr '\0' ' ' < "/proc/${pid}/cmdline"; } 2>/dev/null)
    case "$cmdline" in
        *python3*run.py*)
            kind=dwweb
            tag=run.py
            ;;
        *agent_command*)
            # Pick the agent name (first non-flag argument after agent_command).
            tag=$(echo "$cmdline" | awk '{
                for (i=1; i<=NF; i++) if ($i ~ /agent_command$/) { print $(i+1); exit }
            }')
            kind=agent
            ;;
        *)
            # Fallback: do NOT silently drop the row. Any python3
            # process above the RSS floor is worth recording under
            # ``kind=other`` — the *tag* is a short cmdline slug so an
            # operator scanning the log can still trace it back. Silent
            # drops here are what let the reindex-per-commit and the
            # detached ExtractionMcpHost run invisibly for hours.
            # The RSS floor keeps its original job — silencing processes
            # with neither swap nor RSS — but it must NOT veto a swap
            # holder. A paged-out process has its pages counted in VmSwap
            # and *subtracted* from VmRSS, so judging it by RSS alone
            # discards precisely the evidence we added VmSwap to collect.
            if [ "${rss:-0}" -lt "$OTHER_RSS_FLOOR_KB" ] \
                && [ "${swap:-0}" -lt "$SWAP_FLOOR_KB" ]; then
                # Cheap children (fixtures, tiny helpers) below both
                # floors are the intentional filter — they would spam
                # the log without adding signal.
                unset comm state thr rss peak vmsz swap cmdline etime
                continue
            fi
            kind=other
            # First 40 chars of the cmdline, spaces/quotes squashed
            # (procfs is single-line so newline stripping is defensive).
            tag=$(printf '%s' "$cmdline" | tr -s ' \t\n' '_' | cut -c1-40)
            [ -z "$tag" ] && tag='?'
            ;;
    esac
    # NB: swap_kb is appended LAST, after comm. The tag/comm slug has been
    # field 10 for the whole life of this log (597k+ lines); inserting a
    # column before it would silently turn every existing `awk '{... $10}'`
    # one-liner into a reader of the wrong field. Append-only keeps old
    # analysis snippets and the rows below the RSS floor comparable.
    printf '%s %-6s %-7d %-10s %-10s %-10s %-4s %-2s %-12s %-24s %s\n' \
        "$ts" "$kind" "$pid" "${rss:-?}" "${peak:-?}" "${vmsz:-?}" \
        "${thr:-?}" "${state:-?}" "${etime:-?}" "${tag:-?}" "${swap:-?}"
    unset comm state thr rss peak vmsz swap cmdline tag kind etime
done
