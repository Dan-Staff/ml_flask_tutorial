import random
import unittest

from ml_flask_tutorial.models import LinearRegression


class TestLinearRegression(unittest.TestCase):

    def test_init_default(self):
        clf = LinearRegression()
        self.assertEqual(clf.b0, 0)
        self.assertEqual(clf.b1, 0)

    def test_init(self):
        a, b = 1.2, 1.5
        clf = LinearRegression(a, b)
        self.assertEqual(clf.b0, a)
        self.assertEqual(clf.b1, b)
    
    def test_fit_y_eq_x(self):
        clf = LinearRegression()
        clf.fit(range(5), range(5))
        self.assertEqual(clf.b0, 0.)
        self.assertEqual(clf.b1, 1.)

    def test_fit(self):
        clf = LinearRegression()
        xs = [x for x in range(5)]
        ys = [2.3*x + 1.1 for x in xs]
        clf.fit(xs, ys)
        self.assertAlmostEqual(clf.b0, 1.1)
        self.assertAlmostEqual(clf.b1, 2.3)

    def test_fit_guass(self):
        clf = LinearRegression()
        xs = [x for x in range(5)]
        random.seed(0)
        ys = [2.3*x + 1.1 + random.gauss(0, 0.01) for x in xs]
        clf.fit(xs, ys)
        self.assertAlmostEqual(clf.b0 , 1.1, places=2)
        self.assertAlmostEqual(clf.b1, 2.3, places=2)

if __name__ == "__main__":
    unittest.main()