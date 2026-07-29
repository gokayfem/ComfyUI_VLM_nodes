import { app } from "../../../scripts/app.js";

const LLM_NODE = "PromptGenerateAPI";
const SAFE_SOURCE = "Provider environment variable";
const NO_KEY_SOURCE = "No key (loopback custom endpoint only)";
const SAFE_SOURCES = new Set([SAFE_SOURCE, NO_KEY_SOURCE]);
const CREDENTIAL_WIDGET_INDEX = 2;

function visitGraphNodes(graphData, callback) {
    for (const node of graphData?.nodes ?? []) {
        callback(node);
    }
    for (const subgraph of graphData?.definitions?.subgraphs ?? []) {
        visitGraphNodes(subgraph, callback);
    }
}

function scrubSerializedNode(node) {
    if (node?.type !== LLM_NODE) {
        return;
    }
    const values = node.widgets_values;
    if (Array.isArray(values)) {
        const saved = values[CREDENTIAL_WIDGET_INDEX];
        if (!SAFE_SOURCES.has(saved)) {
            values[CREDENTIAL_WIDGET_INDEX] = SAFE_SOURCE;
        }
        return;
    }
    if (values && typeof values === "object") {
        // Some frontend versions serialize widgets by name.
        delete values.api_key;
        if (!SAFE_SOURCES.has(values.credential_source)) {
            values.credential_source = SAFE_SOURCE;
        }
    }
}

function enforceLiveWidget(node) {
    if (node?.type !== LLM_NODE) {
        return;
    }
    const widget = node.widgets?.find(
        (item) => item.name === "credential_source",
    );
    if (widget && !SAFE_SOURCES.has(widget.value)) {
        widget.value = SAFE_SOURCE;
        widget.callback?.(SAFE_SOURCE);
    }
}

app.registerExtension({
    name: "gokayfem.vlm.api-credential-security",
    async beforeConfigureGraph(graphData) {
        // Runs on the cloned workflow before LiteGraph creates any widgets, so a
        // legacy key never reaches a DOM input or the active graph.
        visitGraphNodes(graphData, scrubSerializedNode);
    },
    loadedGraphNode(node) {
        enforceLiveWidget(node);
    },
});
