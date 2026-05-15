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
# Why no Python-level monitoring: every leak found so far (six of
# them, see the related internal incident analyses) was visible from
# procfs alone. Adding pyrasite / gc.get_objects to a monitoring loop
# on a multi-runtime (tokio + torch + jemalloc) process is a known
# cause of segfaults and is exactly what cost us several agent
# processes during diagnostics.

set -u
exec >>"${HOME}/.deep_worker/logs/rss.log" 2>>"${HOME}/.deep_worker/logs/rss.err"

ts=$(date -Iseconds)

# Emit a header line if the log is freshly rotated (file size < 200 B).
log_file="${HOME}/.deep_worker/logs/rss.log"
mkdir -p "$(dirname "$log_file")"
if [ ! -s "$log_file" ] || [ "$(stat -c %s "$log_file" 2>/dev/null || echo 0)" -lt 200 ]; then
    echo "# format: ts  kind  pid  rss_kb  vmpeak_kb  vmsize_kb  threads  state  etime  comm"
fi

# Find PIDs to monitor: every python3 process whose cmdline contains
# either 'deep_worker_cli.cli.agent_command' or 'run.py' (the dwweb
# entry point). Both filters intentionally narrow — we don't want to
# log every random python script on the box.
for pid in $(pgrep -f 'python3.*run\.py|python3.*deep_worker_cli\.cli\.agent_command' 2>/dev/null); do
    [ -r "/proc/${pid}/status" ] || continue
    # Grab the fields we care about from /proc/PID/status in one read.
    eval "$(awk '
        /^Name:/      { printf "comm=%q\n",   $2 }
        /^State:/     { printf "state=%s\n",  $2 }
        /^Threads:/   { printf "thr=%s\n",    $2 }
        /^VmRSS:/     { printf "rss=%s\n",    $2 }
        /^VmPeak:/    { printf "peak=%s\n",   $2 }
        /^VmSize:/    { printf "vmsz=%s\n",   $2 }
    ' "/proc/${pid}/status" 2>/dev/null)"
    etime=$(ps -o etime= -p "$pid" 2>/dev/null | tr -d ' ')
    [ -z "${rss:-}" ] && continue
    # Identify the role of this process from its cmdline.
    cmdline=$(tr '\0' ' ' < "/proc/${pid}/cmdline" 2>/dev/null)
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
            continue
            ;;
    esac
    printf '%s %-6s %-7d %-10s %-10s %-10s %-4s %-2s %-12s %s\n' \
        "$ts" "$kind" "$pid" "${rss:-?}" "${peak:-?}" "${vmsz:-?}" \
        "${thr:-?}" "${state:-?}" "${etime:-?}" "${tag:-?}"
    unset comm state thr rss peak vmsz cmdline tag kind etime
done
