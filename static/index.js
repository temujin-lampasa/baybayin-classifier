var conv_buttons = document.getElementsByClassName("convLayerBtn")
var fc_buttons = document.getElementsByClassName("fcLayerBtn")

const NUM_LAYERS = 4;
conv_layers = [];
fc_layers = [];

for (let i=1; i<=NUM_LAYERS; i++) {
    conv_layers.push(document.getElementsByClassName("conv-layer"+i))
    fc_layers.push(document.getElementsByClassName("fc-layer"+i))
}


function setBtnOnClick(buttons, layers) {
    for (let i=0; i< NUM_LAYERS; i++){
        buttons[i].onclick = function() {

            // Disable this button. Enable all other buttons.
            for (let j=0; j<NUM_LAYERS; j++) {
                buttons[j].disabled = false;
            }
            buttons[i].disabled = true;

            // Make the layer i visible. Make all other layers invisible.
            for (let j=0; j<NUM_LAYERS; j++) {
                let display = (j!=i) ? "none": ""
                for (let elem of layers[j]) {
                    elem.style.display = display;
                }
            }
        }
    }
}

setBtnOnClick(conv_buttons, conv_layers);
setBtnOnClick(fc_buttons, fc_layers);
