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
    context.lineWidth = '2';
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