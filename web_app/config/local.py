import os
from .default import *

APP_ENV = APP_ENV_LOCAL

SQLALCHEMY_DATABASE_URI = 'sqlite:///'+os.path.join(BASE_DIR, 'database.db')
