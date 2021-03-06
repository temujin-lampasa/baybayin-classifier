from flask import Flask, render_template, url_for, request, redirect, session
from flask_sqlalchemy import SQLAlchemy
from torchvision import transforms
from datetime import datetime
import os
from pathlib import Path
from pprint import PrettyPrinter
import shutil

import simplejson as json
from models import DEFAULT_CNN_PARAMS, DEFAULT_TRAIN_PARAMS, ALTERNATE_CNN_PARAMS, classify_uploaded_file, train_model, generate_feature_maps, generate_character

from forms import CNNForm, FeatureMapsForm, GANForm, NUM_LAYERS
from flask_wtf.csrf import CSRFProtect

# For converting canvas drawing to image
from PIL import Image
import base64
from io import BytesIO

app = Flask(__name__)

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SECRET_KEY'] = '\xd7\x15\xf4\x13k{\xb7b\xfe;D\n\xf3fa7\x9a\x0e\x87q\x0e\x1d\xabV'

csrf = CSRFProtect()
csrf.init_app(app)

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
        # Clear variables when launching for the first time
        session.clear()
        session['cnn_path'] = os.path.join(os.getcwd(), 'default.pt')

        # Create folder for feature maps when launching for the first time
        session['feature_maps_path'] = 'static/feature_maps/0'
        if not os.path.exists('static/feature_maps/0'):
            Path('static/feature_maps/0').mkdir(parents=True, exist_ok=True)
            # session['feature_maps_path'] = 'static/feature_maps/0'

        if not session.get('feature_maps'):
            session['feature_maps'] = os.listdir(session['feature_maps_path'])

        # There should be no gan image when first launching
        session['gan_image'] = None

        FIRST_LAUNCH = False
    
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

    session['gan_images_path'] = f"static/gan/{session['uid']}"
    
    if not os.path.exists(session['gan_images_path']):
        Path(f"static/gan/{session['uid']}").mkdir(parents=True, exist_ok=True)

    if not session.get('train_params'):
        session['train_params'] = DEFAULT_TRAIN_PARAMS
    

    if not session.get('cnn_formdata'):
        layer_params = {k: [v for _ in range(NUM_LAYERS)]  for k, v in DEFAULT_CNN_PARAMS.items()}
        train_params= session['train_params']

        # Set default values for boolean fields.
        all_selected = [i for i in range(NUM_LAYERS)]
        for bool_field in ('conv_layer_on', 'fc_layer_on', 'batch_norm'):
            layer_params[bool_field] = all_selected

        session['cnn_formdata'] = {}
        session['cnn_formdata'].update(layer_params)
        session['cnn_formdata'].update(train_params)
    else:
        session['cnn_formdata'] = json.loads(session['cnn_formdata'])

    if not session.get('gan_formdata'):
        session['gan_formdata'] = {}
    else:
        session['gan_formdata'] = json.loads(session['gan_formdata'])
    # Forms ---------------

    # CNN Layers & Training Form
    cnn_form = CNNForm(data=session['cnn_formdata'])
    # Show Feature Maps
    feature_maps_form = FeatureMapsForm()
    # GAN Form
    gan_form = GANForm()

    formdata = cnn_form.data
    ganformdata = gan_form.data

    if cnn_form.validate_on_submit():
        print("CNN Form validated.")
        formdata.pop('csrf_token')
        session['cnn_formdata'] = json.dumps(formdata, use_decimal=True)
        return redirect('/train')

    else:
        formdata.pop('csrf_token')
        print("CNN Form validation failed")
        session['cnn_formdata'] = json.dumps(formdata, use_decimal=True)
        return render_template("index.html",
                           cnn_form=cnn_form,
                           fm_form=feature_maps_form,
                           gan_form=gan_form)

    if feature_maps_form.validate_on_submit():
        return redirect('/cnn')

    if gan_form.valudate_on_submit():
        print("GAN Form validated.")
        return redirect('/generate')
    else:
        print("GAN Form validation failed")
        return render_template("index.html",
                           cnn_form=cnn_form,
                           fm_form=feature_maps_form,
                           gan_form=gan_form)
    # ---------------------

    return render_template("index.html",
                           cnn_form=cnn_form,
                           fm_form=feature_maps_form,
                           gan_form=gan_form)


@app.route('/train', methods=['GET', 'POST'])
def train():
    # Train here
    # session['train_params'] = request.form
    # TODO: give the params


    # Convert wtforms form data
    # into a format the model can understand
    # session['cnn_formdata'] --> CNN_PARAMS, TRAIN_PARAMS

    cnn_formdata = json.loads(session['cnn_formdata'])
    pp = PrettyPrinter()
    pp.pprint(cnn_formdata)

    # Note: 
    # training forms are single values. 
    # Conv/FC Layer forms are lists of length NUM_LAYERS
    xy_forms = ['kernel', 'stride', 'pool_size']
    fc_forms = ['output_size']
    conv_forms = ['filters', 'kernel', 'stride', 'conv_layer_on', 'pool_size', 'padding']
    train_forms = ['optimizer', 'learning_rate', 'beta1', 'beta2', 'batch_size', 'epochs']


    CNN_params = {
    'batch_norm' : False,
    'dropout' : 0.5,
    'activation_fn' : 'ReLU'
    }

    conv_config_template = {
        'filters': 0,
        'kernel': 0,
        'stride': 0,
        'pool_size': 0,
        'padding': 0,
    }

    fc_config_template = {
        'output_size': 0,
        'dropout': 0
    }
    
    CNN_params['conv_layer_configs'] = [conv_config_template.copy() for _ in range(NUM_LAYERS)]
    CNN_params['fc_layer_configs'] =  [fc_config_template.copy() for _ in range(NUM_LAYERS)]

    TRAIN_params = {
        'epochs': 2,
        'batch_size': 4,
        'optimizer': 'SGD',
        'learning_rate': 0.001,
        'momentum': 0.9,
        'beta1': 0,
        'beta2': 0,
    }

    for layer_idx in range(NUM_LAYERS):
        for field, value in cnn_formdata.items():

            # CONV config
            if field in conv_config_template:
                if field in xy_forms:
                    res = (value[layer_idx]['x'], value[layer_idx]['y'])
                else:
                    res = value[layer_idx]
                CNN_params['conv_layer_configs'][layer_idx][field] = res
            
            # FC config
            if field in fc_config_template:
                res = value[layer_idx]
                CNN_params['fc_layer_configs'][layer_idx][field] = res

            # Train params
            if field in train_forms:
                TRAIN_params[field] = value
                
    pp.pprint(CNN_params)
    pp.pprint(TRAIN_params)

    import dill
    with open(os.path.join(app.root_path, 'test_params.pickle'), 'wb') as file:
        dill.dump(CNN_params, file)
        
    # train_model(
    #     CNN_params,
    #     TRAIN_params,
    #     os.path.join(os.getcwd(), f"users/{session['uid']}/{session['uid']}.pt")
    #     )
    # session['cnn_path'] = os.path.join(os.getcwd(), f"users/{session['uid']}/{session['uid']}.pt")
    return redirect('/')


@app.route('/cnn', methods=['POST', 'GET'])
def cnn():
    # display feature maps here

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

@csrf.exempt
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


# Adapted from: https://stackoverflow.com/questions/41957490/send-canvas-image-data-uint8clampedarray-to-flask-server-via-ajax
# and https://stackoverflow.com/questions/9166400/convert-rgba-png-to-rgb-with-pil
@csrf.exempt
@app.route('/classify_drawing', methods=['GET', 'POST'])
def classify_drawing():
    drawing_path = f"users/{session['uid']}/drawing.jpg"
    if os.path.exists(drawing_path):
        os.remove(drawing_path)
    print(f"Saving drawing to {drawing_path}")

    image_b64 = request.json['image'].split(",")[1]
    image_PIL = Image.open(BytesIO(base64.b64decode(image_b64)))
    image_PIL.load()

    background = Image.new("RGB", image_PIL.size, (255, 255, 255))
    background.paste(image_PIL, mask=image_PIL.split()[3])
    background.save(drawing_path, 'JPEG', quality=80)
    
    with open (drawing_path, 'rb') as d:
        classification, probability = classify_uploaded_file(d, session['cnn_path'])
        # session['drawing_classification'] = classification
        # session['drawing_probability'] = f'{probability*100:.2f}'
    
    response = json.dumps({'status': 'OK', 'class': classification, 'proba': f'{probability*100:.2f}'})
    return response
    

@app.route('/generate', methods=['POST'])
def generate():

    # create new image to avoid caching issues
    if os.listdir(session['gan_images_path']):
        latest_image = sorted(os.listdir(session['gan_images_path']))[-1]
        new_image = f'{int(os.path.splitext(latest_image)[0])+1}.png'
    else:
        new_image = '0.png'
    
    session['gan_character'] = request.form['character']
    session['gan_image'] = os.path.join(session['gan_images_path'], new_image)
    generate_character(session['gan_character'], session['gan_image'])

    print(session['gan_image'])

    return redirect('/')

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8080, debug=True)
