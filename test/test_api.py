import shutil
import unittest

from ml_flask_tutorial.models import LinearRegression
from ml_flask_tutorial.run import create_app


class TestAPI(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.app, cls.db = create_app(testing=True)
        
    @classmethod
    def tearDownClass(cls):
        shutil.rmtree('test_db')

    def setUp(self):
        with self.app.test_request_context():
            if self.db.get('model') is not None:
                del self.db['model']
        
    def test_ls_models(self):
        with self.app.test_client() as context:
            response = context.get('/v1/models')
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json, [])

    def test_model_404(self):
        with self.app.test_client() as context:
            response = context.get('/v1/models/test')
            self.assertEqual(response.status_code, 404)

    def test_post_model(self):

        with self.app.test_client() as context:
            model = LinearRegression(.1, 1.1)
            response = context.post('/v1/models/test', json=model.to_dict())
            self.assertEqual(response.status_code, 200)
            self.assertDictEqual(response.json, model.to_dict())
        
        with self.app.test_client() as context:
            response = context.get('/v1/models/test')
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json, model.to_dict())

    def test_delete(self):
        with self.app.test_request_context():
            self.db['model'] = {}
            self.db['model']['test'] = LinearRegression().to_dict()
        with self.app.test_client() as context:
            response = context.delete('/v1/models/test')
            self.assertEqual(response.status_code, 204)

    def test_fit(self):       
        with self.app.test_request_context():
            self.db['model'] = {}
            self.db['model']['test'] = LinearRegression().to_dict()

        payload = dict(xs=[1,2,3], ys=[2,4,6])
        expected = LinearRegression().fit(payload['xs'], payload['ys']).to_dict()
        with self.app.test_client() as context:
            response = context.put('/v1/models/test/fit', json=payload)
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json, expected)
    
    def test_predict(self):
        with self.app.test_request_context():
            self.db['model'] = {}
            self.db['model']['test'] = LinearRegression(0, 1).to_dict()

        payload = dict(xs=[2,4,6])
        expected = {'ys': LinearRegression(0, 1).predict(payload['xs'])}
        with self.app.test_client() as context:
            response = context.get('/v1/models/test/predict', json=payload)
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json, expected)

if __name__ == "__main__":
    unittest.main()
