from wtforms import widgets
from wtforms.widgets.html5 import NumberInput
from wtforms.fields.html5 import IntegerField, DecimalField
from wtforms.fields import FieldList, FormField, SelectField, SelectMultipleField
from flask_wtf import FlaskForm
from wtforms.validators import InputRequired, NumberRange

from wtforms import (
    DecimalField,
    SelectField,
    BooleanField
)

NUM_LAYERS = 4


class XY_Form(FlaskForm):
    x = IntegerField('x', validators=[InputRequired(), NumberRange(min=0, max=10)])
    y = IntegerField('y', validators=[InputRequired(), NumberRange(min=0, max=10, message="error")])


class MultiCheckboxField(SelectMultipleField):
    widget = widgets.ListWidget(prefix_label=False)
    option_widget = widgets.CheckboxInput()


class CNNForm(FlaskForm):
    filters = FieldList(
        IntegerField('Filters', validators=[InputRequired(), NumberRange(0, 10)]),
        min_entries=NUM_LAYERS
        )

    kernel = FieldList(FormField(XY_Form), min_entries=NUM_LAYERS)

    stride = FieldList(FormField(XY_Form), min_entries=NUM_LAYERS)

    conv_layer_on = MultiCheckboxField(label="On", choices=[(i, f"On-{i}") for i in range(NUM_LAYERS)])

    padding = FieldList(
        SelectField('Padding', choices=["same", "valid"], validators=[InputRequired()]),
        min_entries=NUM_LAYERS
    )

    pool_size = FieldList(FormField(XY_Form), min_entries=NUM_LAYERS)
    batch_norm = MultiCheckboxField(choices=[(i, f"Batch Norm-{i}") for i in range(4)])

    output_size = FieldList(IntegerField('Output Size', validators=[InputRequired()]), min_entries=NUM_LAYERS)
    dropout = FieldList(DecimalField('Dropout', places=1, widget=NumberInput(step=0.1), validators=[InputRequired()]), min_entries=NUM_LAYERS)
    activation = FieldList(SelectField('Activation Function', choices=['sigmoid', 'tanh', 'ReLu'], validators=[InputRequired()]), min_entries=NUM_LAYERS)
    fc_layer_on = MultiCheckboxField(label="On", choices=[(i, f"On-{i}") for i in range(NUM_LAYERS)])


class RetrainModelForm(FlaskForm):
    optimizer = SelectField('Optimizer', choices=['SGD', 'Adam', 'Nadam', 'RMSProp'])
    learning_rate = DecimalField('Dropout', places=2, widget=NumberInput(step=0.01))
    beta1 = DecimalField('Beta 1', places=2, widget=NumberInput(step=0.01))
    beta2 = DecimalField('Beta 2', places=2, widget=NumberInput(step=0.01))
    batch_size = IntegerField('Batch Size')
    epochs = IntegerField('Epochs')
