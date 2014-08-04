import numpy as np
import theano
import theano.tensor as T
from theano import function

class LogisticRegression(object):

    def __init__(self, input, n_in, n_out):
        self.W = theano.shared(np.zeros((n_in, n_out), dtype=theano.config.floatX),
                               name='W')
        self.b = theano.shared(np.zeros((n_out,), dtype=theano.config.floatX),
                               name='b')
        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)

        self.params = [self.W, self.b]

    def negative_log_likelihood(self, y):
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

    def errors(self, y):
        return T.mean(T.neq(self.y_pred, y))

def optimise(data, classes, learning_rate=0.13, n_epochs=5000,
             batch_size=50):

    n_batches = data.shape[0] / batch_size
    n_in = data.shape[1]
    n_out = np.max(classes) + 1

    index = T.lscalar()
    x = T.matrix('x')
    y = T.ivector('y')

    data = theano.shared(np.asarray(data, dtype=theano.config.floatX), borrow=True)
    classes = theano.shared(np.asarray(classes, dtype=theano.config.floatX), borrow=True)
    classes = T.cast(classes, 'int32')

    clf = LogisticRegression(input=x, n_in=n_in, n_out=n_out)

    cost = clf.negative_log_likelihood(y)

    g_W = T.grad(cost, clf.W)
    g_b = T.grad(cost, clf.b)

    updates = [(clf.W, clf.W - learning_rate * g_W),
               (clf.b, clf.b - learning_rate * g_b)]

    train_model = function(
        inputs=[index],
        outputs=cost,
        updates=updates,
        givens={
            x: data[index * batch_size:(index + 1) * batch_size],
            y: classes[index * batch_size:(index + 1) * batch_size],
        }
    )

    test_model = function(
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
            print test_model(minibatch_index)

    return clf
