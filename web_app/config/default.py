import secrets
from os.path import abspath, dirname, join

# Define the application directories
BASE_DIR = dirname(dirname(abspath(__file__)))
MEDIA_DIR = join(BASE_DIR, 'media')

SECRET_KEY = secrets.token_urlsafe(20)

# Database configuration
SQLALCHEMY_TRACK_MODIFICATIONS = False

# App environments
APP_ENV_LOCAL = 'local'
APP_ENV = ''
