import secrets
from os.path import abspath, dirname

# Define the application directory
BASE_DIR = dirname(dirname(abspath(__file__)))

SECRET_KEY = secrets.token_urlsafe(20)

# Database configuration
SQLALCHEMY_TRACK_MODIFICATIONS = False

# App environments
APP_ENV_LOCAL = 'local'
APP_ENV = ''
