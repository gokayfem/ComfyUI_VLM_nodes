import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

const OUTPUT_NAME = "output_text";
const VIEW_TEXT_NODE = "ViewText";
const MODERN_VLM_NODE = "ModernVLM";

function ensureOutputWidget(node) {
    let widget = node.widgets?.find((item) => item.name === OUTPUT_NAME);
    if (!widget) {
        const container = document.createElement("div");
        const header = document.createElement("div");
        const status = document.createElement("span");
        const copy = document.createElement("button");
        const output = document.createElement("textarea");

        status.textContent = "Ready";
        copy.textContent = "Copy";
        copy.type = "button";
        copy.title = "Copy the complete VLM response";
        copy.addEventListener("click", async () => {
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

        output.readOnly = true;
        output.setAttribute("aria-label", "VLM text output");
        header.append(status, copy);
        container.append(header, output);
        Object.assign(container.style, {
            display: "flex",
            flexDirection: "column",
            width: "100%",
            height: "100%",
            minHeight: "150px",
            gap: "6px",
        });
        Object.assign(header.style, {
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
            color: "var(--descrip-text, #aaa)",
            fontSize: "12px",
        });
        Object.assign(copy.style, {
            color: "var(--input-text, #ddd)",
            background: "var(--comfy-input-bg, #202020)",
            border: "1px solid var(--border-color, #555)",
            borderRadius: "5px",
            padding: "3px 9px",
            cursor: "pointer",
        });
        Object.assign(output.style, {
            width: "100%",
            flex: "1",
            minHeight: "120px",
            resize: "vertical",
            boxSizing: "border-box",
            color: "var(--input-text, #ddd)",
            background: "var(--comfy-input-bg, #202020)",
            border: "1px solid var(--border-color, #555)",
            borderRadius: "6px",
            padding: "8px",
            lineHeight: "1.45",
            whiteSpace: "pre-wrap",
        });
        widget = node.addDOMWidget(OUTPUT_NAME, "STRING", container, {
            serialize: false,
            hideOnZoom: false,
        });
        widget.serialize = false;
        widget.inputEl = output;
        widget.statusEl = status;
    }
    return widget;
}

function setOutput(node, text, state = "Complete") {
    const widget = ensureOutputWidget(node);
    const value = Array.isArray(text) ? text.join("\n\n") : String(text ?? "");
    widget.value = value;
    widget.inputEl.value = value;
    if (widget.statusEl) {
        widget.statusEl.textContent = state;
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

function connectedViewTextNodes(source) {
    if (!source?.graph) {
        return [];
    }
    const found = new Set();
    for (const output of source.outputs ?? []) {
        for (const linkId of output.links ?? []) {
            const link = source.graph.links?.get?.(linkId)
                ?? source.graph._links?.get?.(linkId);
            const target = findNode(source.graph, link?.target_id);
            if (target?.type === VIEW_TEXT_NODE) {
                found.add(target);
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
    if (source.type !== MODERN_VLM_NODE) {
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
