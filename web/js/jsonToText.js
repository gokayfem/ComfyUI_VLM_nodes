import { app } from "../../../scripts/app.js";

const OUTPUT_NAME = "formatted_text";

function ensureOutputWidget(node) {
    let widget = node.widgets?.find((item) => item.name === OUTPUT_NAME);
    if (!widget) {
        const output = document.createElement("textarea");
        output.readOnly = true;
        output.setAttribute("aria-label", "Formatted JSON text output");
        Object.assign(output.style, {
            width: "100%",
            height: "100%",
            minHeight: "120px",
            resize: "vertical",
            boxSizing: "border-box",
            color: "var(--input-text, #ddd)",
            background: "var(--comfy-input-bg, #202020)",
            border: "1px solid var(--border-color, #555)",
            borderRadius: "6px",
            padding: "8px",
        });
        widget = node.addDOMWidget(OUTPUT_NAME, "STRING", output, {
            serialize: false,
            hideOnZoom: false,
        });
        widget.serialize = false;
        widget.inputEl = output;
    }
    return widget;
}

app.registerExtension({
    name: "gokayfem.vlm.json-to-text",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "JsonToText") {
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
            const values = Array.isArray(message?.text)
                ? message.text
                : [message?.text ?? ""];
            const widget = ensureOutputWidget(this);
            widget.value = values.join("");
            widget.inputEl.value = widget.value;
            this.setDirtyCanvas?.(true, true);
            return result;
        };
    },
});
