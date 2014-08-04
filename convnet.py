import numpy as np
import theano
import theano.tensor as T
from chordrec import test_data
from theano.tensor.nnet import conv
from theano.tensor.signal import downsample

from logisticregression import LogisticRegression
from mlp import HiddenLayer

DTYPE_FLOATX = theano.config.floatX

class LeNetConvPoolLayer(object):

    def __init__(self, rng, input, image_size, filter_size, n_input_filters,
                 n_filters, pool_size, batch_size):

        image_shape = (batch_size, n_input_filters) + image_size
        filter_shape = (n_filters, n_input_filters) + filter_size

        self.input = input
        fan_in = np.prod(filter_shape)
        W_values = np.asarray(rng.uniform(
            low=-np.sqrt(3.0 / fan_in),
            high=np.sqrt(3.0 / fan_in),
            size=filter_shape), dtype=DTYPE_FLOATX)
        self.W = theano.shared(value=W_values, name='W')

        b_values = np.zeros((n_filters, ), dtype=DTYPE_FLOATX)
        self.b = theano.shared(value=b_values, name='b')

        conv_out = conv.conv2d(input, self.W, filter_shape=filter_shape,
                               image_shape=image_shape)

        pool_out = downsample.max_pool_2d(conv_out, pool_size, ignore_border=True)

        self.output = T.tanh(pool_out + self.b.dimshuffle('x', 0, 'x', 'x'))

        self.params = [self.W, self.b]

def optimise(data, classes,
             n_epochs=1000,
             learning_rate=0.1,
             batch_size=20,
             input_patch_size=(15, 48),
             layer0_filter_size=(4, 5),
             layer0_pool_size=(2, 2),
             layer0_n_filters=10,
             layer1_filter_size=(5, 5),
             layer1_pool_size=(2, 2),
             layer1_n_filters=10,
             layer2_size=100,
):
    n_batches = data.shape[0] / batch_size
    n_classes = np.max(classes) + 1

    index = T.lscalar()
    x = T.matrix('x')
    y = T.ivector('y')

    data = input_to_patches(data, input_patch_size)
    data = theano.shared(np.asarray(data, dtype=DTYPE_FLOATX), borrow=True)
    classes = theano.shared(np.asarray(classes, dtype=DTYPE_FLOATX), borrow=True)
    classes = T.cast(classes, 'int32')

    rng = np.random.RandomState(1)

    layer0_input = x.reshape(
        (batch_size, 1, input_patch_size[0], input_patch_size[1]))

    layer0 = LeNetConvPoolLayer(rng, input=layer0_input,
                                image_size=input_patch_size,
                                filter_size=layer0_filter_size,
                                n_input_filters=1,
                                n_filters=layer0_n_filters,
                                batch_size=batch_size,
                                pool_size=layer0_pool_size)

    layer1_input_size = (
        (np.array(input_patch_size) - layer0_filter_size + 1) / np.array(layer0_pool_size).astype(float))

    assert all(layer1_input_size == layer1_input_size.astype(int))

    layer1 = LeNetConvPoolLayer(rng, input=layer0.output,
                                image_size=tuple(layer1_input_size.astype(int)),
                                filter_size=layer1_filter_size,
                                n_input_filters=layer0_n_filters,
                                n_filters=layer1_n_filters,
                                batch_size=batch_size,
                                pool_size=layer1_pool_size)

    layer2_input_size = np.prod(
        (np.array(layer1_input_size) - layer1_filter_size + 1) / layer1_pool_size) * layer1_n_filters
    assert layer2_input_size == int(layer2_input_size)

    layer2 = HiddenLayer(rng, input=layer1.output.flatten(2),
                         n_in=layer2_input_size,
                         n_out=layer2_size)

    layer3 = LogisticRegression(input=layer2.output,
                                n_in=layer2_size,
                                n_out=n_classes)

    cost = layer3.negative_log_likelihood(y)
    params = layer3.params + layer2.params + layer1.params + layer0.params
    grads = T.grad(cost, params)

    updates = []
    for param, grad in zip(params, grads):
        updates.append((param, param - learning_rate * grad))

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
        outputs=layer3.errors(y),
        givens={
            x: data[index * batch_size:(index + 1) * batch_size],
            y: classes[index * batch_size:(index + 1) * batch_size],
        }
    )

    for epoch in xrange(n_epochs):
        for minibatch_index in xrange(n_batches):
            minibatch_avg_cost = train_model(minibatch_index)
        print epoch, test_model(minibatch_index)
    
def input_to_patches(data, patch_size):
    n_pad = (patch_size[0] - 1) / 2.0
    assert n_pad % 1 == 0

    padding = np.zeros((n_pad, data.shape[1]))
    padded_data = np.vstack((padding, data, padding))

    patches = np.zeros((data.shape[0], patch_size[0] * patch_size[1]))
    for i in xrange(len(data)):
        patches[i, :] = padded_data[i:i + patch_size[0], :].flatten()

    return patches

def get_data():
    data, classes = test_data.get_test_data(1)[0]
    return data, classes
