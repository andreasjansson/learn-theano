import numpy as np
import theano
import theano.tensor as T
from chordrec import test_data

from logisticregression import LogisticRegression

class HiddenLayer(object):

    def __init__(self, rng, input, n_in, n_out, activation=T.tanh):
        self.input = input

        W_values = np.asarray(rng.uniform(
            low=-np.sqrt(6. / (n_in + n_out)),
            high=np.sqrt(6. / (n_in + n_out)),
            size=(n_in, n_out)), dtype=theano.config.floatX)
        if activation == theano.tensor.nnet.sigmoid:
            W_values *= 4

        self.W = theano.shared(value=W_values, name='W')

        b_values = np.zeros((n_out,), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, name='b')

        self.output = activation(T.dot(input, self.W) + self.b)
        self.params = [self.W, self.b]

class MLP(object):

    def __init__(self, rng, input, n_in, n_hidden, n_out):

        self.hidden_layer = HiddenLayer(
            rng=rng, input=input, n_in=n_in, n_out=n_hidden,
            activation=T.tanh)

        self.output_layer = LogisticRegression(
            input=self.hidden_layer.output, n_in=n_hidden, n_out=n_out)

        self.L1 = (abs(self.hidden_layer.W).sum() +
                   abs(self.output_layer.W).sum())
        self.L2_sqr = (abs(self.hidden_layer.W ** 2).sum() +
                       abs(self.output_layer.W ** 2).sum())

        self.negative_log_likelihood = self.output_layer.negative_log_likelihood
        self.errors = self.output_layer.errors

        self.params = self.hidden_layer.params + self.output_layer.params

def optimise(data, classes, learning_rate=0.03, L1_reg=0, L2_reg=0.0001,
             n_epochs=2000, batch_size=50, n_hidden=50):

    n_batches = data.shape[0] / batch_size
    n_in = data.shape[1]
    n_out = np.max(classes) + 1

    index = T.lscalar()
    x = T.matrix('x')
    y = T.ivector('y')

    data = theano.shared(np.asarray(data, dtype=theano.config.floatX), borrow=True)
    classes = theano.shared(np.asarray(classes, dtype=theano.config.floatX), borrow=True)
    classes = T.cast(classes, 'int32')

    rng = np.random.RandomState(1)

    clf = MLP(rng=rng, input=x, n_in=n_in, n_hidden=n_hidden, n_out=n_out)

    cost = (clf.negative_log_likelihood(y) +
            L1_reg * clf.L1 + L2_reg * clf.L2_sqr)

    gparams = []
    for param in clf.params:
        gparam = T.grad(cost, param)
        gparams.append(gparam)

    updates = []
    for param, gparam in zip(clf.params, gparams):
        updates.append((param, param - learning_rate * gparam))

    train_model = theano.function(
        inputs=[index],
        outputs=cost,
        updates=updates,
        givens={
            x: data[index * batch_size:(index + 1) * batch_size],
            y: classes[index * batch_size:(index + 1) * batch_size],
        }
    )

    test_model = theano.function(
        inputs=[index],
        outputs=clf.errors(y),
        givens={
            x: data[index * batch_size:(index + 1) * batch_size],
            y: classes[index * batch_size:(index + 1) * batch_size],
        }
    )

    for epoch in xrange(n_epochs):
        for minibatch_index in xrange(n_batches):
            minibatch_avg_cost = train_model(minibatch_index)
        if epoch % 100 == 0:
            print epoch, test_model(minibatch_index)

def get_data():
    data, classes = test_data.get_test_data(1)[0]
    return data, classes
