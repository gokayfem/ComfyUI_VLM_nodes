import { app } from "../../../scripts/app.js";

function firstValue(value) {
    return Array.isArray(value) && value.length === 1 ? value[0] : value;
}

app.registerExtension({
    name: "gokayfem.vlm.play-music",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "PlayMusic") {
            return;
        }
        const onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = async function (message) {
            onExecuted?.apply(this, arguments);
            const mode = firstValue(this.widgets?.[0]?.value) ?? "always";
            if (mode === "on empty queue" && (app.ui?.lastQueueSize ?? 0) > 0) {
                return;
            }

            const raw = firstValue(message?.a);
            const samples = Array.isArray(raw?.[0]) ? raw[0] : raw;
            const sampleRate = Number(firstValue(message?.b));
            if (!samples?.length || !Number.isFinite(sampleRate)) {
                return;
            }

            this.__vlmAudioSource?.stop?.();
            const AudioContext = window.AudioContext ?? window.webkitAudioContext;
            this.__vlmAudioContext ??= new AudioContext({ sampleRate });
            await this.__vlmAudioContext.resume();
            const buffer = this.__vlmAudioContext.createBuffer(
                1,
                samples.length,
                sampleRate,
            );
            buffer.getChannelData(0).set(samples);
            const source = this.__vlmAudioContext.createBufferSource();
            const gain = this.__vlmAudioContext.createGain();
            gain.gain.value = Number(firstValue(this.widgets?.[1]?.value) ?? 0.5);
            source.buffer = buffer;
            source.connect(gain);
            gain.connect(this.__vlmAudioContext.destination);
            source.start();
            this.__vlmAudioSource = source;
        };

        const onRemoved = nodeType.prototype.onRemoved;
        nodeType.prototype.onRemoved = function (...args) {
            this.__vlmAudioSource?.stop?.();
            void this.__vlmAudioContext?.close?.();
            this.__vlmAudioSource = null;
            this.__vlmAudioContext = null;
            return onRemoved?.apply(this, args);
        };
    },
});
