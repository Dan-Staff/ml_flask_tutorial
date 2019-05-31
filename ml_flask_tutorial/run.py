import os
import shutil

from flask import Flask, abort, jsonify, request
from flaskext.zodb import ZODB

from ml_flask_tutorial.models import LinearRegression, DeepThought

def load_model(db, payload):
    if payload.get('type') == 'LinearRegression':
        new_model = LinearRegression().from_dict(payload)
    elif payload.get('type') == 'DeepThought':
        new_model = DeepThought().from_dict(payload)
    else:
        new_model = None
    return new_model

def create_app(testing=False):
    """Create a new flask application using ZODB as a lightweight backend

    ZODB is a lightweight database which can be treated like a python dictionary.
    Note: A real application would use a more fully featured database
    
    :param testing: set True when unit testing, defaults to False
    :type testing: bool, optional
    :return: database handle and app instance tuple
    """
    app = Flask(__name__)

    if testing:
        if os.path.isdir('test_db'):
            shutil.rmtree('test_db')
        os.mkdir('test_db')
        app.config['ZODB_STORAGE'] = 'file://test_db/test.fs'
        app.config['TESTING'] = True
    else:
        if os.path.isdir('db'):
            shutil.rmtree('db')
        os.mkdir('db')
        app.config['ZODB_STORAGE'] = 'file://db/app.fs'
    
    db = ZODB(app)

    @app.route('/v1/models')
    def models():
        return jsonify([model for model in db.get('model', [])])

    @app.route('/v1/models/<name>', methods=['POST', 'GET', 'DELETE'])
    def linear_regression(name):
        models = db.get('model', None)
        if models is None:
            db['model'] = {}

        if request.method == 'POST':
            payload = request.json
            new_model = load_model(db, payload)
            if new_model is None:
                return abort(400)
            db['model'][name] = new_model.to_dict()
            return jsonify(new_model.to_dict())

        if request.method == 'GET':
            model = db.get('model', {}).get(name)
            if model is None:
                abort(404)
            return jsonify(model)

        if request.method == 'DELETE':
            if name is None:
                abort(404)    
            model = db['model'][name]
            del db['model'][name]
            return ('', 204)

    @app.route('/v1/models/<name>/fit', methods=['PUT'])
    def fit_model(name):
        kwargs = db.get('model', {}).get(name)
        if kwargs is None:
            abort(404)
        
        model = load_model(db, kwargs)
        if model is None:
            return abort(400)
        xs = request.json.get('xs', [])
        ys = request.json.get('ys', [])
        model.fit(xs, ys)
        db['model'][name] = model.to_dict()
        return jsonify(model.to_dict())

    @app.route('/v1/models/<name>/predict', methods=['GET'])
    def predict_model(name):
        kwargs = db.get('model', {}).get(name)
        if kwargs is None:
            abort(404)
        
        model = load_model(db, kwargs)
        if model is None:
            return abort(400)
        xs = request.json['xs']
        ys = model.predict(xs)
        return jsonify({'ys': ys})

    return app, db

if __name__ == '__main__':

    app, db = create_app()
    app.run()
