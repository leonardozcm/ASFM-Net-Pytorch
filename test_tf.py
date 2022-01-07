import tensorflow as tf
import torch
from torch import nn

class RMSELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, yhat, y):
        loss_feat = torch.sqrt(torch.sum(torch.pow(torch.subtract(yhat,y),2),1,keepdim=False))
        loss_feat = torch.mean(loss_feat,dim=0, keepdim=False)
        return loss_feat

def mlp_conv(inputs, layer_dims, bn=None, bn_params=None):
    for i, num_out_channel in enumerate(layer_dims[:-1]):
        inputs = tf.contrib.layers.conv1d(
            inputs, num_out_channel,
            kernel_size=1,
            normalizer_fn=bn,
            normalizer_params=bn_params,
            scope='conv_%d' % i)
    outputs = tf.contrib.layers.conv1d(
        inputs, layer_dims[-1],
        kernel_size=1,
        activation_fn=None,
        scope='conv_%d' % (len(layer_dims) - 1))
    return outputs

def point_maxpool(inputs, npts, keepdims=False):
    outputs = [tf.reduce_max(f, axis=1, keepdims=keepdims)
        for f in tf.split(inputs, npts, axis=1)]
    print(outputs[0].shape)
    print(len(outputs))
    return tf.concat(outputs, axis=0)


def point_unpool(inputs, npts):
    inputs = tf.split(inputs, inputs.shape[0], axis=0)
    outputs = [tf.tile(f, [1, npts[i], 1]) for i,f in enumerate(inputs)]
    return tf.concat(outputs, axis=1)

def test_encoder():
    bs = 32
    inputs = tf.constant(0.0, shape=(bs, 4096, 3))
    npts=tf.constant(4096, shape=[32,], name='num_points')
    print(npts.shape)

    inputs_1 = tf.reshape(inputs,[1,-1,3])
    print(inputs_1.shape)

    features = mlp_conv(inputs_1, [128, 256])
    print("features",features.shape)
    p_maxpool=point_maxpool(features, npts, keepdims=True)
    print("p_maxpool",p_maxpool.shape)
    features_global = point_unpool(p_maxpool,npts)
    print("features_global",features_global.shape)
    # print(tf.split(inputs_1, npts, axis=1)[0].shape)

bs = 32

def test_loss(arr1,arr2):
    model1_l2 = tf.convert_to_tensor(arr1)
    model2_l2 = tf.convert_to_tensor(arr2)
    print(model1_l2.shape)
    loss_feat = tf.sqrt(tf.reduce_sum(tf.pow(tf.subtract(model1_l2, model2_l2), 2), 1, keepdims=False))
    loss_feat = tf.reduce_mean(loss_feat, keepdims=False)
    with tf.Session():
        # We can also use 'c.eval()' here.
        print(loss_feat.eval())

def test_torch_loss(arr1,arr2):
    model1_l2 = torch.from_numpy(arr1)
    model2_l2 = torch.from_numpy(arr2)
    loss_feat = torch.sqrt(torch.sum(torch.pow(torch.subtract(model1_l2,model2_l2),2),1,keepdim=False))
    loss_feat = torch.mean(loss_feat,dim=0, keepdim=False)
    print(loss_feat)

import numpy as np
arr1=np.random.rand(bs,1024)
arr2=np.random.rand(bs,1024)
test_loss(arr1,arr2)
test_torch_loss(arr1, arr2)
