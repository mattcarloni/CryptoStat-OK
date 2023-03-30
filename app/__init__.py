from flask import Flask
from .extentions import mongo
from .main import main
from .auth import auth
from .crypto import crypto
from .twitter import twitter


def create_app(config_object="app.settings"):
    app = Flask(__name__)
    app.config.from_object(config_object)
    mongo.init_app(app)

    app.register_blueprint(main)
    app.register_blueprint(auth)
    app.register_blueprint(crypto)
    app.register_blueprint(twitter)

    return app
