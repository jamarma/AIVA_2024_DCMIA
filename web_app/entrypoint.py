import os
from app import create_app
from flask import send_from_directory

settings_module = os.getenv('APP_SETTINGS_MODULE')
app = create_app(settings_module)


@app.route('/media/<filename>')
def media_results(filename):
    dir_path = app.config['MEDIA_DIR']
    return send_from_directory(dir_path, filename, mimetype='application/png')
