import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

const OUTPUT_NAME = "output_text";
const VIEW_TEXT_NODE = "ViewText";
const STREAMING_SOURCE_NODES = new Set([
    "ModernVLM",
    "Moondream31Query",
    "Moondream31Caption",
    "PromptGenerateAPI",
    "HostedVLMAPI",
    "VLMVideoTemporalReasoner",
]);

function textMetrics(value) {
    const text = String(value ?? "");
    const words = text.trim() ? text.trim().split(/\s+/u).length : 0;
    const lines = text ? text.split("\n").length : 0;
    return `${text.length.toLocaleString()} chars · ${words.toLocaleString()} words · ${lines.toLocaleString()} lines`;
}

function makeButton(label, title, handler) {
    const button = document.createElement("button");
    button.textContent = label;
    button.type = "button";
    button.title = title;
    button.addEventListener("click", handler);
    Object.assign(button.style, {
        color: "var(--input-text, #ddd)",
        background: "var(--comfy-input-bg, #202020)",
        border: "1px solid var(--border-color, #555)",
        borderRadius: "5px",
        padding: "3px 8px",
        cursor: "pointer",
        whiteSpace: "nowrap",
    });
    return button;
}

function ensureOutputWidget(node) {
    let widget = node.widgets?.find((item) => item.name === OUTPUT_NAME);
    if (widget) {
        return widget;
    }

    const container = document.createElement("div");
    const header = document.createElement("div");
    const status = document.createElement("span");
    const meta = document.createElement("span");
    const actions = document.createElement("div");
    const output = document.createElement("textarea");

    status.textContent = "Ready";
    meta.textContent = textMetrics("");
    output.readOnly = true;
    output.wrap = "soft";
    output.spellcheck = false;
    output.setAttribute("aria-label", "VLM text output");

    const copy = makeButton("Copy", "Copy complete text", async () => {
        const previous = copy.textContent;
        try {
            await navigator.clipboard.writeText(output.value);
            copy.textContent = "Copied";
        } catch {
            copy.textContent = "Copy failed";
        }
        window.setTimeout(() => {
            copy.textContent = previous;
        }, 1200);
    });
    const download = makeButton("Save", "Download output as a UTF-8 text file", () => {
        const blob = new Blob([output.value], {
            type: "text/plain;charset=utf-8",
        });
        const url = URL.createObjectURL(blob);
        const anchor = document.createElement("a");
        anchor.href = url;
        anchor.download = `vlm-output-${new Date().toISOString().replaceAll(":", "-")}.txt`;
        anchor.click();
        URL.revokeObjectURL(url);
    });
    const wrap = makeButton("Wrap: on", "Toggle long-line wrapping", () => {
        const enabled = output.wrap !== "off";
        output.wrap = enabled ? "off" : "soft";
        output.style.whiteSpace = enabled ? "pre" : "pre-wrap";
        output.style.overflowX = enabled ? "auto" : "hidden";
        wrap.textContent = enabled ? "Wrap: off" : "Wrap: on";
    });
    const follow = makeButton("Follow: on", "Follow streaming output", () => {
        widget.followOutput = !widget.followOutput;
        follow.textContent = widget.followOutput ? "Follow: on" : "Follow: off";
    });

    actions.append(wrap, follow, copy, download);
    header.append(status, meta, actions);
    container.append(header, output);

    Object.assign(container.style, {
        display: "flex",
        flexDirection: "column",
        width: "100%",
        height: "100%",
        minHeight: "190px",
        gap: "6px",
    });
    Object.assign(header.style, {
        display: "grid",
        gridTemplateColumns: "auto minmax(0, 1fr) auto",
        alignItems: "center",
        gap: "9px",
        color: "var(--descrip-text, #aaa)",
        fontSize: "11px",
    });
    Object.assign(meta.style, {
        overflow: "hidden",
        textOverflow: "ellipsis",
        whiteSpace: "nowrap",
    });
    Object.assign(actions.style, {
        display: "flex",
        gap: "4px",
        justifyContent: "flex-end",
    });
    Object.assign(output.style, {
        width: "100%",
        flex: "1",
        minHeight: "160px",
        resize: "vertical",
        boxSizing: "border-box",
        color: "var(--input-text, #ddd)",
        background: "var(--comfy-input-bg, #202020)",
        border: "1px solid var(--border-color, #555)",
        borderRadius: "6px",
        padding: "9px",
        lineHeight: "1.45",
        whiteSpace: "pre-wrap",
        overflowWrap: "anywhere",
        tabSize: "4",
    });

    widget = node.addDOMWidget(OUTPUT_NAME, "STRING", container, {
        serialize: false,
        hideOnZoom: false,
    });
    widget.serialize = false;
    widget.inputEl = output;
    widget.statusEl = status;
    widget.metaEl = meta;
    widget.followOutput = true;
    return widget;
}

function setOutput(node, text, state = "Complete") {
    const widget = ensureOutputWidget(node);
    const value = Array.isArray(text) ? text.join("\n\n") : String(text ?? "");
    widget.value = value;
    widget.inputEl.value = value;
    widget.statusEl.textContent = state;
    widget.metaEl.textContent = textMetrics(value);
    if (widget.followOutput && state === "Streaming…") {
        widget.inputEl.scrollTop = widget.inputEl.scrollHeight;
    }
    node.setDirtyCanvas?.(true, true);
}

function findNode(graph, id) {
    if (!graph || id == null) {
        return null;
    }
    return graph.getNodeById?.(id)
        ?? graph.getNodeById?.(String(id))
        ?? graph.getNodeById?.(Number(id))
        ?? null;
}

function linkFor(graph, linkId) {
    return graph?.links?.get?.(linkId)
        ?? graph?._links?.get?.(linkId)
        ?? null;
}

function isReroute(node) {
    return String(node?.type ?? "").toLowerCase().includes("reroute");
}

function connectedViewTextNodes(source) {
    if (!source?.graph) {
        return [];
    }
    const found = new Set();
    const visited = new Set([source.id]);
    const queue = [source];
    while (queue.length) {
        const current = queue.shift();
        for (const output of current.outputs ?? []) {
            for (const linkId of output.links ?? []) {
                const link = linkFor(source.graph, linkId);
                const target = findNode(source.graph, link?.target_id);
                if (!target || visited.has(target.id)) {
                    continue;
                }
                visited.add(target.id);
                if (target.type === VIEW_TEXT_NODE) {
                    found.add(target);
                } else if (isReroute(target)) {
                    queue.push(target);
                }
            }
        }
    }
    return [...found];
}

function updateFromProgress({ nodeId, text }) {
    const source = findNode(app.rootGraph ?? app.graph, nodeId);
    if (!source) {
        return;
    }
    if (source.type === VIEW_TEXT_NODE) {
        setOutput(source, text, "Streaming…");
        return;
    }
    if (!STREAMING_SOURCE_NODES.has(source.type)) {
        return;
    }
    for (const target of connectedViewTextNodes(source)) {
        setOutput(target, text, "Streaming…");
    }
}

app.registerExtension({
    name: "gokayfem.vlm.view-text",
    async setup() {
        api.addEventListener("progress_text", ({ detail }) => {
            updateFromProgress(detail ?? {});
        });
    },
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== VIEW_TEXT_NODE) {
            return;
        }
        const onCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function (...args) {
            const result = onCreated?.apply(this, args);
            ensureOutputWidget(this);
            if (Array.isArray(this.size)) {
                this.setSize?.([
                    Math.max(this.size[0], 430),
                    Math.max(this.size[1], 290),
                ]);
            }
            return result;
        };
        const onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            const result = onExecuted?.apply(this, arguments);
            setOutput(this, message?.text, "Complete");
            return result;
        };
    },
    onNodeOutputsUpdated(nodeOutputs) {
        for (const [nodeId, output] of Object.entries(nodeOutputs ?? {})) {
            const node = findNode(app.rootGraph ?? app.graph, nodeId);
            if (node?.type === VIEW_TEXT_NODE && output?.text != null) {
                setOutput(node, output.text, "Complete");
            }
        }
    },
});
