from wtforms.widgets.html5 import NumberInput
from wtforms.fields.html5 import IntegerField, DecimalField
from wtforms.fields import FieldList, FormField, SelectField
from wtforms import Form, validators

from wtforms import (
    DecimalField,
    SelectField,
    BooleanField
)

class XY_Form(Form):
    x = IntegerField('x', [validators.DataRequired()])
    y = IntegerField('y', [validators.DataRequired()])

class CNNForm(Form):
    filters = IntegerField('filters')
    kernel = FormField(XY_Form)
    stride = FormField(XY_Form)

    conv_layer_on = BooleanField('conv_on')
    padding = SelectField('Padding:', choices=["same", "valid"])
    pool_size = FormField(XY_Form)
    batch_norm = BooleanField('Batch Norm')

    output_size = IntegerField('output_size')
    dropout = DecimalField('dropout', places=1, widget=NumberInput(step=0.1))
    activation = SelectField('activation', choices=['sigmoid', 'tanh', 'ReLu'])
    dense_layer_on = BooleanField('dense_on')
