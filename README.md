# ml_flask_tutorial

## Introduction

This repository is meant to be an educational exercise that highlights some pain points when trying to package machine learning models behind REST APIs.

`run.py` contains a REST interface for a 2D linear regression model build using Flask.
`models.py` contains the linear regression code.

ZODB is used as a database because of its ease of use. ZODB can be treated like a persistent python dictionary.

## Installation

Note this project uses *PYTHON 2*, not Python 3.

Using conda or a virtual environment is highly recommended.

To clone this project
```bash
git clone https://github.com/Dan-Staff/ml_flask_tutorial.git
```

To install project dependencies
```bash
cd ml_flask_tutorial
pip install -e .
```

### Running the server

Execture run.py and the system will list on `localhost` on port `5000`.

```bash
python ml_flask_tutorial/run.py
```

### Interacting with the server

In a seperate terminal you can send requests to the server. e.g. using `curl`

```bash
curl http://127.0.0.1:5000/v1/models
```

## Exercises

### Exercise 1
Instead of using `curl` make some requests to the server using python. For example using the `requests` library.
Make a least one meaningful request to each endpoint in `run.py` from a python REPL or script.

### Exercise 2
Run all of the unittests from the root directory using
```
python2 -m unittest discover -v -s ./test -p "test_*.py"
```

One of the tests is failing. Can you fix it?

### Exercise 3
Look in `models.py`. There is a class called DeepThought.

Part a: DeepThought takes a very long time to train. What problems do you forsee interacting with this class using the REST interface?

Part b: DeepThought is represented by a very long uninteligable string of numbers. What problems do you forsee storing this data in the database and interacting with it using the REST API. Can you think of a more suitable way of storing the model (e.g. using major cloud providers)?



