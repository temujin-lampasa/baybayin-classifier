from flask import Flask, render_template, url_for, request, redirect, session
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import io
import os

# from models import DEFAULT_CNN_PARAMS as BAYBAYIN_DEFAULT_CNN

# from models import baybayin_net, pre_process_image, classes
# from PIL import Image
# import torch
# import torch.nn.functional as F


app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SECRET_KEY'] = '\xd7\x15\xf4\x13k{\xb7b\xfe;D\n\xf3fa7\x9a\x0e\x87q\x0e\x1d\xabV'

db = SQLAlchemy(app)

DEFAULT_TRAIN_PARAMS = {
    'optimizer': 'SGD',
    'learning_rate': 0.00,
    'momentum': 0.00,
    'beta1': 0.00,
    'beta2': 0.00,
    'batch_size': 32,    
    'epochs': 1,
}
DEFAULT_CNN_PARAMS = {
    'filters': 1,
    'kernel_x': 1,
    'kernel_y': 1,
    'stride_x': 1,
    'stride_y': 1,
    'padding': "valid",
    'pool_x':1,
    'pool_y': 1,
    'output_size': 1,
    'dropout': 1.0,
    'activation': "sigmoid",
 }


class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    date_created = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f'<User {id}>'


@app.route('/about')
def about():
    return render_template("about.html")

@app.route('/', methods=['GET', 'POST'])
def index():
    if not session.get('train_params'):
        session['train_params'] = DEFAULT_TRAIN_PARAMS
    
    print("Current train params")
    print(session['train_params'])

    if not session.get('cnn_params'):
        session['cnn_params'] = DEFAULT_CNN_PARAMS

    if not session.get('uid'):
        new_user = User()
        db.session.add(new_user)
        db.session.commit()
        session['uid'] = new_user.id

        # Each user has a unique directory
        # Dir. name is user ID
        if not os.path.exists('users'):
            os.mkdir('users')
        if not os.path.exists(f"users/{session['uid']}"):
            os.mkdir(f"users/{session['uid']}")

    return render_template("index.html")


@app.route('/train', methods=['POST'])
def train():
    # Train here
    session['train_params'] = request.form
    train_model(session['train_params'])
    return redirect('/')


@app.route('/cnn', methods=['POST'])
def cnn():
    # run CNN here
    session['cnn_params'] = request.form
    get_feature_maps(session['cnn_params'])
    return redirect('/')


# @app.route('/classify', methods=['POST'])
# def classify():
#     # classify image here
#     if request.files["drawing"]:
#         drawing = request.files["drawing"]
#         drawing = drawing.read()
#         drawing = pre_process_image(Image.open(io.BytesIO(drawing))).unsqueeze(0)
#         classifier = baybayin_net(BAYBAYIN_DEFAULT_CNN)
#         classifier.load_state_dict(torch.load("default.pt"))
#         print(classifier)
#         predictions = F.softmax(classifier(drawing), 1)
#         probability, class_index = predictions.max(1)
#         classification = classes[class_index.item()]
#         print("Prediction:", classification)
#         print("probability:", probability.item())
#         session['classification'] = classification
#         session['probability'] = f'{probability.item()*100:.2f}'
#     else:
#         if session.get("classification"):
#             del session["classification"]
#         if session.get("probability"):
#             del session["probability"]
#     return redirect('/')

def get_feature_maps(args):
    print("Getting feature maps...")
    print(args)

def train_model(args):
    print("Training...")
    print(args)

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8080, debug=True)
