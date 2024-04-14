from flask import Flask
from flask_login import LoginManager

from app.db import db

login_manager = LoginManager()


def create_app(settings_module):
    app = Flask(__name__, instance_relative_config=True)
    app.config.from_object(settings_module)
    app.config.from_pyfile('config.py', silent=True)

    db.init_app(app)

    login_manager.login_view = 'auth.login'
    login_manager.init_app(app)

    from .auth import auth_bp
    app.register_blueprint(auth_bp)

    return app
