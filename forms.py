from wtforms import widgets
from wtforms.widgets.html5 import NumberInput
from wtforms.fields.html5 import IntegerField, DecimalField
from wtforms.fields import FieldList, FormField, SelectField, SelectMultipleField, HiddenField
from flask_wtf import FlaskForm
from wtforms.validators import InputRequired, NumberRange

from wtforms import (
    DecimalField,
    SelectField,
    BooleanField
)

NUM_LAYERS = 4

class BaseForm(FlaskForm):
    def __init__(self, *args, **kwargs):
        kwargs['csrf_enabled'] = False
        super().__init__(*args, **kwargs)

class XY_Form(BaseForm):
    x = IntegerField('x', validators=[InputRequired(), NumberRange(min=0, max=10)])
    y = IntegerField('y', validators=[InputRequired(), NumberRange(min=0, max=10)])


class MultiCheckboxField(SelectMultipleField):
    widget = widgets.ListWidget(prefix_label=False)
    option_widget = widgets.CheckboxInput()


class CNNForm(FlaskForm):
    filters = FieldList(
        IntegerField('Filters', validators=[InputRequired(), NumberRange(0, 10)]),
        min_entries=NUM_LAYERS
        )

    kernel = FieldList(FormField(XY_Form, label="Kernel"), min_entries=NUM_LAYERS)
    stride = FieldList(FormField(XY_Form, label="Stride"), min_entries=NUM_LAYERS)

    conv_layer_on = MultiCheckboxField(label="On", choices=[(i, "On") for i in range(NUM_LAYERS)], coerce=int)

    padding = FieldList(
        SelectField('Padding', choices=["same", "valid"], validators=[InputRequired()]),
        min_entries=NUM_LAYERS
    )

    pool_size = FieldList(FormField(XY_Form, label="Pool Size"), min_entries=NUM_LAYERS)
    batch_norm = MultiCheckboxField(choices=[(i, "Batch Norm") for i in range(4)], coerce=int)

    output_size = FieldList(IntegerField('Output Size', validators=[InputRequired()]), min_entries=NUM_LAYERS)
    dropout = FieldList(DecimalField('Dropout', places=1, widget=NumberInput(step=0.1), validators=[InputRequired()]), min_entries=NUM_LAYERS)
    activation = FieldList(SelectField('Activation Function', choices=['sigmoid', 'tanh', 'ReLu'], validators=[InputRequired()]), min_entries=NUM_LAYERS)
    fc_layer_on = MultiCheckboxField(label="On", choices=[(i, "On") for i in range(NUM_LAYERS)], coerce=int)

    optimizer = SelectField('Optimizer', choices=['SGD', 'Adam', 'Nadam', 'RMSProp'])
    learning_rate = DecimalField('Learning Rate', places=3, widget=NumberInput(step=0.01))
    beta1 = DecimalField('Beta 1', places=2, widget=NumberInput(step=0.01))
    beta2 = DecimalField('Beta 2', places=2, widget=NumberInput(step=0.01))
    batch_size = IntegerField('Batch Size')
    epochs = IntegerField('Epochs')


class FeatureMapsForm(FlaskForm):
    h = HiddenField()
