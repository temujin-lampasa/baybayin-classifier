var conv_buttons = document.getElementsByClassName("convLayerBtn")
var fc_buttons = document.getElementsByClassName("fcLayerBtn")

for (let btn of conv_buttons) {
    btn.onclick = function() {
        for (let btn2 of conv_buttons) {
            btn2.disabled = false;
        }
        btn.disabled = true;
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


