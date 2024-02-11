import { app } from "/scripts/app.js";
import { ComfyWidgets } from "/scripts/widgets.js";

app.registerExtension({
    name: "n.JsonToText",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        
        if (nodeData.name === "JsonToText") {
            console.warn("JsonToText");
            
            const onExecuted = nodeType.prototype.onExecuted;
            
            nodeType.prototype.onExecuted = function (message) {
                if (this.widgets) {
					for (let i = 1; i < this.widgets.length; i++) {
						this.widgets[i].onRemove?.();
					}
					this.widgets.length = 1;
				}
                
                // Call the original onExecuted method if it exists.
                onExecuted?.apply(this, arguments);
                
                // Check if the "text" widget already exists.
                let textWidget = this.widgets.find(w => w.name === "newtext");
                if (!textWidget) {
                    // If the "text" widget does not exist, create it.
                    textWidget = ComfyWidgets["STRING"](this, "newtext", ["STRING", { multiline: true }], app).widget;
                }
                
                // Generate a random number and set it as the value of the "text" widget.
                
                textWidget.inputEl.readOnly = true;
                textWidget.inputEl.style.opacity = 0.6;
                textWidget.value = message["text"].join(""); 
                // change color of the widget
                console.log(message)    

            };
        }
    },
});