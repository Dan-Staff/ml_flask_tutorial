from setuptools import setup, find_packages

setup(
    name='ml_flask_tutorial',
    version='0.0.1',
    url='https://github.com/Dan-Staff/ml_flask_tutorial.git',
    author='Daniel Staff',
    author_email='daniel.r.staff@gmail.com',
    description='An example of using flask to wrap a predictive model',
    packages=find_packages(),    
    install_requires=['Flask >= 1.0.2', 'Flask-ZODB >= 0.1'],
)
