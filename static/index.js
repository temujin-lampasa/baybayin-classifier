var conv_buttons = document.getElementsByClassName("convLayerBtn");
var fc_buttons = document.getElementsByClassName("fcLayerBtn");

const NUM_LAYERS = 4;
conv_layers = [];
fc_layers = [];

for (let i=1; i<=NUM_LAYERS; i++) {
    conv_layers.push(document.getElementsByClassName("conv-layer"+i));
    fc_layers.push(document.getElementsByClassName("fc-layer"+i));
}


function set_layer_button(buttons, layers) {
    for (let i=0; i< NUM_LAYERS; i++){
        buttons[i].onclick = function() {

            // Disable this button. Enable all other buttons.
            for (let j=0; j<NUM_LAYERS; j++) {
                buttons[j].disabled = false;
            }
            buttons[i].disabled = true;

            // Make the layer i visible. Make all other layers invisible.
            for (let j=0; j<NUM_LAYERS; j++) {
                let display = (j!=i) ? "none": "";
                for (let elem of layers[j]) {
                    elem.style.display = display;
                }
            }
        }
    }
}

set_layer_button(conv_buttons, conv_layers);
set_layer_button(fc_buttons, fc_layers);


// Set first buttons clicked by default
conv_buttons[0].click();
fc_buttons[0].click();


conv_on_buttons = [];
fc_on_buttons = [];

conv_fields = [];
fc_fields = [];


for (let i=0; i<NUM_LAYERS; i++){
    conv_on_buttons.push(document.getElementById("conv_layer_on-"+i));
    fc_on_buttons.push(document.getElementById("fc_layer_on-"+i));
}

for (let i=1; i<=NUM_LAYERS; i++){
    conv_fields.push(document.querySelectorAll("div.conv-layer"+i+" input," +"div.conv-layer"+i+" select"));
    fc_fields.push(document.querySelectorAll("div.fc-layer"+i+" input," +"div.fc-layer"+i+" select"));
}

function disable_fields(layer_type, layer_num){
    if (layer_type=="conv"){
        input_fields = conv_fields;
    } else {
        input_fields = fc_fields;
    }
    for (let field of input_fields[layer_num]){
        if (field.name != (layer_type+"_layer_on")){
            field.disabled= !field.disabled;
        }
    }
}


function set_on_button(button_list, layer_type){
    for (let i=0; i<NUM_LAYERS; i++){
        button_list[i].onclick = function(){
            // Disable all fields in its layer
            disable_fields(layer_type, i);
        }
    }
}


set_on_button(conv_on_buttons, "conv");
set_on_button(fc_on_buttons, "fc");