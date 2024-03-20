import { app } from "../../../scripts/app.js";

app.registerExtension({
    name: "n.PlayMusic",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "PlayMusic") {
            console.warn("PlayMusic");
            const onExecuted = nodeType.prototype.onExecuted;
            nodeType.prototype.onExecuted = async function () {
                onExecuted?.apply(this, arguments);

                // Check for "on empty queue" condition, if applicable
                if (this.widgets[0].value === "on empty queue") {
                    if (app.ui.lastQueueSize !== 0) {
                        await new Promise((r) => setTimeout(r, 500));
                    }
                    if (app.ui.lastQueueSize !== 0) {
                        return;
                    }
                }
                
                // Assuming that 'arguments[0].a' is the waveform and 'arguments[0].b' is the sample rate
                let waveform = arguments[0].a; // An array of floats (-1 to 1)
                let sampleRate = arguments[0].b; // The sample rate of the audio
                console.log(waveform, sampleRate);
                // Create AudioContext
                let audioCtx = new (window.AudioContext || window.webkitAudioContext)({sampleRate: sampleRate});
                
                // Create AudioBuffer
                let buffer = audioCtx.createBuffer(1, waveform[0].length, sampleRate);
                
                // Fill the AudioBuffer
                buffer.getChannelData(0).set(waveform[0]);
                
                // Create a source and connect it to the buffer
                let source = audioCtx.createBufferSource();
                source.buffer = buffer;
                source.connect(audioCtx.destination);
                
                // Set volume, if applicable. Assuming the volume is the second widget's value.
                let volume = this.widgets[1].value; 
                if (volume !== undefined) {
                    let gainNode = audioCtx.createGain();
                    gainNode.gain.value = volume;
                    source.connect(gainNode);
                    gainNode.connect(audioCtx.destination);
                }
                
                // Play the sound
                source.start();
            };
        }
    },
});