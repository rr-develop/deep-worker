// Deep-Worker-Web - Agent Stats page (§6.2).
//
// Renders the JSON from GET /api/agents/<name>/stats. ALL data is written
// via textContent / DOM API — never innerHTML with server data — so agent
// names / model names / values cannot inject markup (XSS-safe).
//
// No polling: the page renders the snapshot captured at fetch time; the
// user can click "🔄 Refresh" to re-fetch manually.

"use strict";

function agentNameFromUrl() {
    const params = new URLSearchParams(window.location.search);
    return params.get("agent");
}

function el(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
}

function setText(id, text) {
    const node = document.getElementById(id);
    if (node) node.textContent = text;
}

// Format a number with thousands separators; null/undefined → "n/a".
function fmt(value) {
    if (value === null || value === undefined) return "n/a";
    if (typeof value === "number") {
        return value.toLocaleString("en-US");
    }
    return String(value);
}

// Format a ratio (0..1) as a percent; null → "n/a".
function fmtPct(value) {
    if (value === null || value === undefined) return "n/a";
    if (typeof value !== "number") return String(value);
    return (value * 100).toFixed(1) + "%";
}

function totalsRows(totals) {
    const rows = [
        ["Requests", fmt(totals.requests)],
        ["Requests with usage", fmt(totals.requests_with_usage)],
        ["Usage coverage", fmtPct(totals.usage_coverage)],
        ["Prompt tokens", fmt(totals.prompt_tokens)],
        ["Completion tokens", fmt(totals.completion_tokens)],
        ["Total tokens", fmt(totals.total_tokens)],
        ["Cache hit tokens", fmt(totals.cache_hit_tokens)],
        ["Cache miss tokens", fmt(totals.cache_miss_tokens)],
        ["Cache ratio (total, observed)", fmtPct(totals.cache_ratio_total_observed)],
        ["Cache ratio (total, lower bound)", fmtPct(totals.cache_ratio_total_lower_bound)],
        ["Cache ratio (total, upper bound)", fmtPct(totals.cache_ratio_total_upper_bound)],
        ["Cache ratio (prompt, observed)", fmtPct(totals.cache_ratio_prompt_observed)],
        ["Cache coverage", fmtPct(totals.cache_coverage)],
        ["Avg tokens per usage request", fmt(totals.avg_tokens_per_usage_observed_request)],
        ["Avg tokens per usage response", fmt(totals.avg_tokens_per_usage_observed_response)],
        ["Has cache", totals.has_cache ? "yes" : "no"],
    ];
    return rows;
}

function buildTotalsTable(totals) {
    const table = document.createElement("table");
    const tbody = document.createElement("tbody");
    for (const [label, value] of totalsRows(totals)) {
        const tr = document.createElement("tr");
        tr.appendChild(el("td", null, label));
        const td = el("td", "num", value);
        if (value === "n/a") td.classList.add("n-a");
        tr.appendChild(td);
        tbody.appendChild(tr);
    }
    table.appendChild(tbody);
    return table;
}

function buildContextTable(context) {
    if (!context || !context.snapshots) return null;
    const table = document.createElement("table");
    const tbody = document.createElement("tbody");
    const rows = [
        ["Snapshots", fmt(context.snapshots)],
        ["Latest context used", fmt(context.latest_context_used)],
        ["Latest context size", fmt(context.latest_context_size)],
        ["Max context used", fmt(context.max_context_used)],
        ["Max context size", fmt(context.max_context_size)],
    ];
    for (const [label, value] of rows) {
        const tr = document.createElement("tr");
        tr.appendChild(el("td", null, label));
        const td = el("td", "num", value);
        if (value === "n/a") td.classList.add("n-a");
        tr.appendChild(td);
        tbody.appendChild(tr);
    }
    table.appendChild(tbody);
    return table;
}

function renderByProvider(container, byProvider) {
    if (!byProvider || byProvider.length === 0) {
        container.appendChild(el("p", null, "No provider records."));
        return;
    }
    for (const provider of byProvider) {
        container.appendChild(el("h3", "section-title", "Provider: " + provider.provider));
        container.appendChild(buildTotalsTable(provider.totals));

        const ctxTable = buildContextTable(provider.totals.context);
        if (ctxTable) {
            container.appendChild(el("h4", "section-title", "Context snapshots (grok fallback)"));
            container.appendChild(ctxTable);
        }

        if (provider.by_model && provider.by_model.length > 0) {
            container.appendChild(el("h4", "section-title", "By model"));
            for (const modelAgg of provider.by_model) {
                container.appendChild(el("p", null, "Model: " + modelAgg.model));
                container.appendChild(buildTotalsTable(modelAgg.totals));
            }
        }
    }
}

// Last successfully rendered stats snapshot. The Copy button copies from
// this; it is null (and the button disabled) until a successful load.
let lastStats = null;

function setCopyEnabled(enabled) {
    const btn = document.getElementById("copy-btn");
    if (btn) btn.disabled = !enabled;
}

// Human-readable plain-text rendering of the stats snapshot, mirroring the
// on-page tables (labels/values come from the same rows builders).
function statsToText(data) {
    const lines = [];
    lines.push("Agent Stats: " + (data.agent || "n/a"));
    lines.push([
        "Generated: " + (data.generated_at || "n/a"),
        "Skipped records: " + fmt(data.skipped_records),
        "Duplicate records: " + fmt(data.duplicate_records),
        "Partial tail: " + (data.partial_tail ? "yes" : "no"),
    ].join(" · "));
    lines.push("");

    lines.push("Totals");
    for (const [label, value] of totalsRows(data.totals || {})) {
        lines.push("  " + label + ": " + value);
    }

    const byProvider = data.by_provider || [];
    lines.push("");
    if (byProvider.length === 0) {
        lines.push("By provider");
        lines.push("  No provider records.");
    }
    for (const provider of byProvider) {
        lines.push("");
        lines.push("Provider: " + provider.provider);
        for (const [label, value] of totalsRows(provider.totals)) {
            lines.push("  " + label + ": " + value);
        }
        const context = provider.totals && provider.totals.context;
        if (context && context.snapshots) {
            lines.push("  Context snapshots (grok fallback):");
            lines.push("    Snapshots: " + fmt(context.snapshots));
            lines.push("    Latest context used: " + fmt(context.latest_context_used));
            lines.push("    Latest context size: " + fmt(context.latest_context_size));
            lines.push("    Max context used: " + fmt(context.max_context_used));
            lines.push("    Max context size: " + fmt(context.max_context_size));
        }
        if (provider.by_model && provider.by_model.length > 0) {
            lines.push("  By model:");
            for (const modelAgg of provider.by_model) {
                lines.push("    Model: " + modelAgg.model);
                for (const [label, value] of totalsRows(modelAgg.totals)) {
                    lines.push("      " + label + ": " + value);
                }
            }
        }
    }
    return lines.join("\n");
}

// Copy the last rendered stats to the clipboard as plain text.
async function copyStats() {
    if (!lastStats) return;
    try {
        await navigator.clipboard.writeText(statsToText(lastStats));
        const btn = document.getElementById("copy-btn");
        if (btn) {
            const original = btn.textContent;
            btn.textContent = "✅ Copied";
            window.setTimeout(() => { btn.textContent = original; }, 1500);
        }
    } catch (error) {
        console.error("copy failed", error);
    }
}

function renderStats(data) {
    const container = document.getElementById("stats-content");
    container.replaceChildren();

    lastStats = data;
    setCopyEnabled(true);

    setText("stats-agent", "Agent Stats: " + data.agent);
    const meta = [
        "Generated: " + (data.generated_at || "n/a"),
        "Skipped records: " + fmt(data.skipped_records),
        "Duplicate records: " + fmt(data.duplicate_records),
        "Partial tail: " + (data.partial_tail ? "yes" : "no"),
    ].join(" · ");
    setText("stats-meta", meta);

    const totalsTitle = el("h3", "section-title", "Totals");
    container.appendChild(totalsTitle);
    container.appendChild(buildTotalsTable(data.totals || {}));

    const ctxTable = buildContextTable((data.totals || {}).context);
    if (ctxTable) {
        container.appendChild(el("h4", "section-title", "Context snapshots (grok fallback)"));
        container.appendChild(ctxTable);
    }

    container.appendChild(el("h3", "section-title", "By provider"));
    renderByProvider(container, data.by_provider);
}

function renderError(message) {
    const container = document.getElementById("stats-content");
    container.replaceChildren();
    lastStats = null;
    setCopyEnabled(false);
    const panel = el("div", "error-panel", "Failed to load stats: " + message);
    container.appendChild(panel);
}

async function loadStats() {
    const name = agentNameFromUrl();
    if (!name) {
        renderError("Missing ?agent=<name> query parameter.");
        return;
    }
    setText("stats-agent", "Agent Stats: " + name);
    const content = document.getElementById("stats-content");
    content.replaceChildren();
    content.appendChild(el("p", "loading", "Loading stats..."));
    try {
        const response = await fetch(
            "/api/agents/" + encodeURIComponent(name) + "/stats",
            { cache: "no-store" }
        );
        if (!response.ok) {
            let detail = "HTTP " + response.status;
            try {
                const err = await response.json();
                if (err && err.detail) detail = String(err.detail);
            } catch (_e) {
                // keep the status-based message
            }
            renderError(detail);
            return;
        }
        const data = await response.json();
        renderStats(data);
    } catch (error) {
        renderError(String(error));
    }
}

document.addEventListener("DOMContentLoaded", () => {
    document.getElementById("copy-btn").addEventListener("click", () => {
        copyStats();
    });
    document.getElementById("refresh-btn").addEventListener("click", () => {
        loadStats();
    });
    loadStats();
});
