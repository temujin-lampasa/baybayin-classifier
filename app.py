from flask import Flask, render_template, url_for, request, redirect, session
from flask_sqlalchemy import SQLAlchemy
from torchvision import transforms
from datetime import datetime
import os
from pathlib import Path
import shutil

from models import DEFAULT_CNN_PARAMS, DEFAULT_TRAIN_PARAMS, ALTERNATE_CNN_PARAMS, classify_uploaded_file, train_model, generate_feature_maps

from forms import CNNForm, RetrainModelForm, NUM_LAYERS
from flask_wtf.csrf import CSRFProtect

app = Flask(__name__)

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SECRET_KEY'] = '\xd7\x15\xf4\x13k{\xb7b\xfe;D\n\xf3fa7\x9a\x0e\x87q\x0e\x1d\xabV'

CSRFProtect(app)

db = SQLAlchemy(app)

FIRST_LAUNCH = True

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
    global FIRST_LAUNCH
    if FIRST_LAUNCH:
        # Clear display variables and reset CNN path when launching for the first time
        if session.get('classification'):
            del session['classification']

        if session.get('probability'):
            del session['probability']

        if session.get('feature_maps'):
            del session['feature_maps']

        session['cnn_path'] = os.path.join(os.getcwd(), 'default.pt')

        # Create folder for feature maps when launching for the first time
        session['feature_maps_path'] = 'static/feature_maps/0'
        if not os.path.exists('static/feature_maps/0'):
            Path('static/feature_maps/0').mkdir(parents=True, exist_ok=True)
            session['feature_maps_path'] = 'static/feature_maps/0'
    
        if not session.get('feature_maps'):
            session['feature_maps'] = os.listdir(session['feature_maps_path'])
        FIRST_LAUNCH = False

    if not session.get('train_params'):
        session['train_params'] = DEFAULT_TRAIN_PARAMS
    
    if not session.get('cnn_params'):
        session['cnn_params'] = DEFAULT_CNN_PARAMS


    # Forms ---------------

    # Set default values
    # Repeat data for each field 1 type per layer
    cnn_formdata = {k: [v for _ in range(NUM_LAYERS)]  for k, v in session['cnn_params'].items()}
    all_selected = [i for i in range(NUM_LAYERS)]
    cnn_formdata['conv_layer_on'] = all_selected
    cnn_formdata['fc_layer_on'] = all_selected
    cnn_formdata['batch_norm'] = all_selected

    cnn_form = CNNForm(data=cnn_formdata)
    retrain_form = RetrainModelForm(data=session['train_params'])

    if cnn_form.validate_on_submit():
        print("Form validated.")
        # ERROR: Object of type 'decimal' is not JSON serializable
        # session['cnn_params'] = cnn_form.data
        return redirect('/cnn')
    else:
        print("Form validation failed")
        print(cnn_form.data)

    if retrain_form.validate_on_submit():
        print("Train form validated.")
        return redirect("/")
    # ---------------------

    if not session.get('uid'):
        new_user = User()
        db.session.add(new_user)
        db.session.commit()
        session['uid'] = new_user.id

    # temporary workaround to user folder
    if not os.path.exists(os.path.join(os.getcwd(), 'users')):
        os.mkdir(os.path.join(os.getcwd(), 'users'))

    if not os.path.exists(os.path.join(os.getcwd(), f"users/{session['uid']}")):
        os.mkdir(os.path.join(os.getcwd(), f"users/{session['uid']}"))

    return render_template("index.html",
                           cnn_form=cnn_form,
                           retrain_form=retrain_form)


@app.route('/train', methods=['POST'])
def train():
    # Train here
    # session['train_params'] = request.form

    # retrain mode with alternate hyperparameters
    train_model(
        ALTERNATE_CNN_PARAMS,
        DEFAULT_TRAIN_PARAMS,
        os.path.join(os.getcwd(), f"users/{session['uid']}/{session['uid']}.pt")
        )
    session['cnn_path'] = os.path.join(os.getcwd(), f"users/{session['uid']}/{session['uid']}.pt")
    return redirect('/')


@app.route('/cnn', methods=['POST', 'GET'])
def cnn():
    # run CNN here

    # use the current CNN
    print(session['cnn_path'])

    # create a new folder to address caching issues
    if os.listdir(session['feature_maps_path']):
        shutil.rmtree(session['feature_maps_path'])
        split_path = os.path.split(session['feature_maps_path'])
        session['feature_maps_path'] = f'{split_path[0]}/{int(split_path[-1])+1}'
        print(session['feature_maps_path'])
        if not os.path.isdir(session['feature_maps_path']):
            Path(session['feature_maps_path']).mkdir(parents=True, exist_ok=True)
    
    # use a path for saving feature maps
    print(session['feature_maps_path'])

    # generate and save feature maps
    session['feature_maps'] = generate_feature_maps(session['feature_maps_path'], session['cnn_path'])
    session['feature_maps'] = [os.path.join(session['feature_maps_path'], filepath) for filepath in session['feature_maps']]
    return redirect('/')


@app.route('/classify', methods=['POST'])
def classify():
    print(session['cnn_path'])
    # classify image here
    if request.files["drawing"]:
        drawing = request.files["drawing"]
        classification, probability = classify_uploaded_file(drawing, session['cnn_path'])
        session['classification'] = classification
        session['probability'] = f'{probability*100:.2f}'
    else:
        # clear classification and probability if no file uploaded
        if session.get("classification"):
            del session["classification"]
        if session.get("probability"):
            del session["probability"]
    return redirect('/')

# def get_feature_maps(args):
    # print("Getting feature maps...")
    # print(args)
    # print(session['cnn_path'])
    # generate_feature_maps(None, session['cnn_path'])

# def train_model(args):
#     print("Training...")
#     print(args)

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8080, debug=True)
