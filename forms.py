from wtforms.widgets.html5 import NumberInput
from wtforms.fields.html5 import IntegerField
from wtforms.fields import FieldList, FormField, StringField
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

    on = BooleanField('On')
    padding = SelectField('Padding:', choices=["same", "valid"])
    pool = FormField(XY_Form)
    batch_norm = BooleanField('Batch Norm')
