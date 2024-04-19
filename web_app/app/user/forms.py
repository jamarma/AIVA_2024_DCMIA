from wtforms import Form
from wtforms import FileField
from flask_wtf.file import FileRequired


class UploadForm(Form):
    """Represents a form to upload a file."""
    image = FileField('Browse files', [
        FileRequired()
    ])
