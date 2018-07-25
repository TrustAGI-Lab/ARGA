import tensorflow as tf
import numpy as np
from model import ARGA, ARVGA, Discriminator
from optimizer import OptimizerAE, OptimizerVAE
import scipy.sparse as sp
from input_data import load_data
import inspect
from preprocessing import preprocess_graph, sparse_to_tuple, mask_test_edges, construct_feed_dict
flags = tf.app.flags
FLAGS = flags.FLAGS

def get_placeholder(adj):
    placeholders = {
        'features': tf.sparse_placeholder(tf.float32),
        'adj': tf.sparse_placeholder(tf.float32),
        'adj_orig': tf.sparse_placeholder(tf.float32),
        'dropout': tf.placeholder_with_default(0., shape=()),
        'real_distribution': tf.placeholder(dtype=tf.float32, shape=[adj.shape[0], FLAGS.hidden2],
                                            name='real_distribution')

    }

    return placeholders


def get_model(model_str, placeholders, num_features, num_nodes, features_nonzero):
    discriminator = Discriminator()
    d_real = discriminator.construct(placeholders['real_distribution'])
    model = None
    if model_str == 'arga_ae':
        model = ARGA(placeholders, num_features, features_nonzero)

    elif model_str == 'arga_vae':
        model = ARVGA(placeholders, num_features, num_nodes, features_nonzero)

    return d_real, discriminator, model


def format_data(data_name):
    # Load data

    adj, features, y_test, tx, ty, test_maks, true_labels = load_data(data_name)


    # Store original adjacency matrix (without diagonal entries) for later
    adj_orig = adj
    adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
    adj_orig.eliminate_zeros()

    adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges(adj)
    adj = adj_train

    if FLAGS.features == 0:
        features = sp.identity(features.shape[0])  # featureless

    # Some preprocessing
    adj_norm = preprocess_graph(adj)

    num_nodes = adj.shape[0]

    features = sparse_to_tuple(features.tocoo())
    num_features = features[2][1]
    features_nonzero = features[1].shape[0]

    pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
    norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)

    adj_label = adj_train + sp.eye(adj_train.shape[0])
    adj_label = sparse_to_tuple(adj_label)
    items = [adj, num_features, num_nodes, features_nonzero, pos_weight, norm, adj_norm, adj_label, features, true_labels, train_edges, val_edges, val_edges_false, test_edges, test_edges_false, adj_orig]
    feas = {}
    for item in items:
        # item_name = [ k for k,v in locals().iteritems() if v == item][0]
        feas[retrieve_name(item)] = item


    return feas

def get_optimizer(model_str, model, discriminator, placeholders, pos_weight, norm, d_real,num_nodes):
    if model_str == 'arga_ae':
        d_fake = discriminator.construct(model.embeddings, reuse=True)
        opt = OptimizerAE(preds=model.reconstructions,
                          labels=tf.reshape(tf.sparse_tensor_to_dense(placeholders['adj_orig'],
                                                                      validate_indices=False), [-1]),
                          pos_weight=pos_weight,
                          norm=norm,
                          d_real=d_real,
                          d_fake=d_fake)
    elif model_str == 'arga_vae':
        opt = OptimizerVAE(preds=model.reconstructions,
                           labels=tf.reshape(tf.sparse_tensor_to_dense(placeholders['adj_orig'],
                                                                       validate_indices=False), [-1]),
                           model=model, num_nodes=num_nodes,
                           pos_weight=pos_weight,
                           norm=norm,
                           d_real=d_real,
                           d_fake=discriminator.construct(model.embeddings, reuse=True))
    return opt

def update(model, opt, sess, adj_norm, adj_label, features, placeholders, adj):
    # Construct feed dictionary
    feed_dict = construct_feed_dict(adj_norm, adj_label, features, placeholders)
    feed_dict.update({placeholders['dropout']: FLAGS.dropout})

    feed_dict.update({placeholders['dropout']: 0})
    emb = sess.run(model.z_mean, feed_dict=feed_dict)

    z_real_dist = np.random.randn(adj.shape[0], FLAGS.hidden2)
    feed_dict.update({placeholders['real_distribution']: z_real_dist})

    for j in range(5):
        _, reconstruct_loss = sess.run([opt.opt_op, opt.cost], feed_dict=feed_dict)
    d_loss, _ = sess.run([opt.dc_loss, opt.discriminator_optimizer], feed_dict=feed_dict)
    g_loss, _ = sess.run([opt.generator_loss, opt.generator_optimizer], feed_dict=feed_dict)

    avg_cost = reconstruct_loss

    return emb, avg_cost


def retrieve_name(var):
    callers_local_vars = inspect.currentframe().f_back.f_locals.items()
    return [var_name for var_name, var_val in callers_local_vars if var_val is var][0]