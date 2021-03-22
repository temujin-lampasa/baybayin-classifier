from wtforms.widgets.html5 import NumberInput
from wtforms.fields.html5 import IntegerField, DecimalField
from wtforms.fields import FieldList, FormField, SelectField
from flask_wtf import FlaskForm
from wtforms.validators import InputRequired, NumberRange

from wtforms import (
    DecimalField,
    SelectField,
    BooleanField
)

class XY_Form(FlaskForm):
    x = IntegerField('x', validators=[InputRequired(), NumberRange(min=0, max=10)])
    y = IntegerField('y', validators=[InputRequired(), NumberRange(min=0, max=10, message="error")])

class CNNForm(FlaskForm):
    filters = IntegerField('filters', validators=[InputRequired(), NumberRange(0, 10)])
    kernel = FormField(XY_Form)
    stride = FormField(XY_Form)

    conv_layer_on = BooleanField('conv_on')
    padding = SelectField('Padding:', choices=["same", "valid"], validators=[InputRequired()])
    pool_size = FormField(XY_Form)
    batch_norm = BooleanField('Batch Norm')

    output_size = IntegerField('output_size', validators=[InputRequired()])
    dropout = DecimalField('dropout', places=1, widget=NumberInput(step=0.1), validators=[InputRequired()])
    activation = SelectField('activation', choices=['sigmoid', 'tanh', 'ReLu'], validators=[InputRequired()])
    dense_layer_on = BooleanField('dense_on')


class RetrainModelForm(FlaskForm):
    optimizer = SelectField('optimizer', choices=['SGD', 'Adam', 'Nadam', 'RMSProp'])
    learning_rate = DecimalField('dropout', places=2, widget=NumberInput(step=0.01))
    beta1 = DecimalField('beta1', places=2, widget=NumberInput(step=0.01))
    beta2 = DecimalField('beta2', places=2, widget=NumberInput(step=0.01))
    batch_size = IntegerField('batch_size')
    epochs = IntegerField('epochs')
