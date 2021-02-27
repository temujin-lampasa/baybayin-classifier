from flask import Flask, render_template, url_for, request, redirect, session
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import os

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SECRET_KEY'] = '\xd7\x15\xf4\x13k{\xb7b\xfe;D\n\xf3fa7\x9a\x0e\x87q\x0e\x1d\xabV'

db = SQLAlchemy(app)

DEFAULT_CNN_PARAMS = { }
DEFAULT_TRAIN_PARAMS = { }


class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    date_created = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f'<User {id}>'


@app.route('/', methods=['GET', 'POST'])
def index():

    if not session.get('train_params'):
        session['train_params'] = DEFAULT_TRAIN_PARAMS

    if not session.get('cnn_params'):
        session['cnn_params'] = DEFAULT_CNN_PARAMS

    if not session.get('uid'):
        new_user = User()
        db.session.add(new_user)
        db.session.commit()
        session['uid'] = new_user.id

        # Each user has a unique directory
        # Dir. name is user ID
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


def get_feature_maps(args):
    print("Getting feature maps...")
    print(args)

def train_model(args):
    print("Training...")
    print(args)

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8080, debug=True)