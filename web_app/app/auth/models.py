from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash

from app.db import db, BaseModelMixin


class User(UserMixin, db.Model, BaseModelMixin):
    __tablename__ = "users"

    id = db.Column(db.Integer, primary_key=True)
    first_name = db.Column(db.String(15), nullable=False)
    last_name = db.Column(db.String(15), nullable=False)
    email = db.Column(db.String(40), unique=True, nullable=False)
    username = db.Column(db.String(15), unique=True, nullable=False)
    password = db.Column(db.String(102), nullable=False)

    def __init__(self, first_name, last_name, email, username, password):
        self.first_name = first_name
        self.last_name = last_name
        self.email = email
        self.username = username
        self.password = generate_password_hash(password)

    def verify_password(self, password):
        return check_password_hash(self.password, password)

    def update_password(self, password):
        self.password = generate_password_hash(password)

    @staticmethod
    def get_by_username(username):
        return User.query.filter_by(username=username).first()

    @staticmethod
    def get_by_email(email):
        return User.query.filter_by(email=email).first()
