//Source: obtained from: https://developer.mozilla.org/en-US/docs/Web/API/Element/mouseup_event
//Date: 3/26/2021
//Name of Author/Programmer: MDN Contributors. Contributors by commit history can be found here: https://github.com/mdn/content/commits/main/files/en-us/web/api/element/mouseup_event/index.html
const drawingCanvas = document.getElementById('drawing-canvas');
const context = drawingCanvas.getContext('2d');

let drawing = false;
let x = 0;
let y = 0;

function draw(context, x1, y1, x2, y2){
    context.beginPath();
    context.strokeStyle = 'black';
    context.lineWidth = '5';
    context.moveTo(x1, y1);
    context.lineTo(x2, y2);
    context.stroke();
}

drawingCanvas.addEventListener('mousedown', e =>{
    drawing = true;
    x = e.offsetX;
    y = e.offsetY;
})

drawingCanvas.addEventListener('mouseup', e => {
    draw(context, x, y, e.offsetX, e.offsetY);
    x = e.offsetX;
    y = e.offsetY;
    drawing = false;
})

drawingCanvas.addEventListener('mousemove', e =>{
    if (drawing)
        draw(context, x, y, e.offsetX, e.offsetY);
        x = e.offsetX;
        y = e.offsetY;
})


const drawing_pred = document.getElementById("drawing-pred");
const drawing_proba = document.getElementById("drawing-proba");

// Saving canvas image
const submitButton = document.querySelectorAll(".drawing-section .drawing-button")[1]
submitButton.onclick = function() {
    img_base64 = drawingCanvas.toDataURL();
    console.log("Saving image...")
    data = {
        "image": img_base64
    };
    $.ajax( "/classify_drawing", {
        type: 'POST',
        data: JSON.stringify(data),
        contentType: 'application/json;charset=UTF-8',
        success: function(response) {
            response = JSON.parse(response);
            drawing_pred.innerHTML = response['class'];
            drawing_proba.innerHTML = response['proba'] + '%';
        }
    });
}