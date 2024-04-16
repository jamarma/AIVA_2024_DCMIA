from wtforms import Form
from wtforms import FileField
from flask_wtf.file import FileRequired
from wtforms import validators


class UploadForm(Form):
    image = FileField('Browse files', [
        FileRequired()
    ])
