var conv_buttons = document.getElementsByClassName("convLayerBtn")
var fc_buttons = document.getElementsByClassName("fcLayerBtn")

const NUM_LAYERS = 4;
layers = [];

for (let i=1; i<=NUM_LAYERS; i++) {
    layers.push(document.getElementsByClassName("cnn-layer"+i))
}


for (let i=0; i< NUM_LAYERS; i++){
    conv_buttons[i].onclick = function() {

        // Disable this button. Enable all other buttons.
        for (let j=0; j<NUM_LAYERS; j++) {
            conv_buttons[j].disabled = false;
        }
        conv_buttons[i].disabled = true;

        // Make the layer i visible. Make all other layers invisible.
        for (let j=0; j<NUM_LAYERS; j++) {
            let display = (j!=i) ? "none": ""
            for (let elem of layers[j]) {
                elem.style.display = display;
            }
        }
    }
}

for (let btn of fc_buttons) {
    btn.onclick = function() {
        for (let btn2 of fc_buttons) {
            btn2.disabled = false;
        }
        btn.disabled = true;
    }
}


