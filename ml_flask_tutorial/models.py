import time
import random

def mean(xs):
    return sum(xs) / len(xs)

class LinearRegression:

    def __init__(self, b0=0., b1=0.):
        """Create a new Linear Regression Model
        
        :param b0: gradient parameter, defaults to 0.
        :type b0: float, optional
        :param b1: intersection parameter, defaults to 0.
        :type b1: float] optional
        """
        self.b0 = b0
        self.b1 = b1

    def __repr__(self):
        return "LinearRegression({}, {})".format(self.b0, self.b1)

    def fit(self, xs, ys):
        """Estimate the cooeficients of the linear regression model
        
        :param xs: list of dependent variables x
        :type xs: [float]
        :param ys: list of response values y
        :type ys: [float]
        """

        n = len(xs)
        if not len(ys) == n:
            raise ValueError(
                "{} and {} must contain the same number of variables".format("xs", "ys")
            )
        
        m_xs, m_ys = mean(xs), mean(ys)

        SS_xy = sum(y*x for x, y in zip(xs, ys)) - n*m_ys*m_xs 
        SS_xx = sum(x*x for x in xs) - n*m_xs*m_xs

        self.b1 = SS_xy / SS_xx 
        self.b0 = m_ys - self.b1*m_xs

        return self

    def predict(self, xs):
        """Estimate response y for each dependent variable x 
        
        :param xs: list of dependent variables
        :type xs: [float]
        :return: list of response values
        :rtype: [float]
        """
        return [x*self.b1 + self.b0 for x in xs]

    def to_dict(self):
        return {'b0': self.b0, 'b1': self.b1, 'type': 'LinearRegression'}
    
    @classmethod
    def from_dict(cls, kwargs):
        return LinearRegression(kwargs['b0'], kwargs['b1'])


class DeepThought:

    def __init__(self):
        pass

    def __repr__(self):
        return 'DeepThought()'

    def fit(self, xs, ys):
        time.sleep(7.5*1000000)
        self
    
    def predict(self, xs):
        return 42
    
    def to_dict(self):
        return {
            'type': 'DeepThought',
            'data': ''.join(str(random.randint(0, 9)) for _ in range(1e6))
        }
    
    @classmethod
    def from_dict(cls, kwargs):
        return DeepThought()



    