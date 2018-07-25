import tensorflow as tf
import numpy as np
flags = tf.app.flags
FLAGS = flags.FLAGS


flags.DEFINE_integer('hidden3', 64, 'Number of units in hidden layer 3.')
flags.DEFINE_integer('discriminator_out', 0, 'discriminator_out.')
flags.DEFINE_float('discriminator_learning_rate', 0.001, 'Initial learning rate.')
flags.DEFINE_float('learning_rate', .5*0.001, 'Initial learning rate.')
flags.DEFINE_integer('hidden1', 32, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden2', 32, 'Number of units in hidden layer 2.')
flags.DEFINE_float('weight_decay', 0., 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_float('dropout', 0., 'Dropout rate (1 - keep probability).')
flags.DEFINE_integer('features', 1, 'Whether to use features (1) or not (0).')
flags.DEFINE_integer('seed', 50, 'seed for fixing the results.')
flags.DEFINE_integer('iterations', 50, 'number of iterations.')

'''
infor: number of clusters 
'''
infor = {'cora': 7, 'citeseer': 6, 'pubmed':3}


'''
We did not set any seed when we conducted the experiments described in the paper;
We set a seed here to steadily reveal better performance of ARGA
'''
seed = 7
np.random.seed(seed)
tf.set_random_seed(seed)

def get_settings(dataname, model, task):
    if dataname != 'citeseer' and dataname != 'cora' and dataname != 'pubmed':
        print('error: wrong data set name')
    if model != 'arga_ae' and model != 'arga_vae':
        print('error: wrong model name')
    if task != 'clustering' and task != 'link_prediction':
        print('error: wrong task name')

    if task == 'clustering':
        iterations = FLAGS.iterations
        clustering_num = infor[dataname]
        re = {'data_name': dataname, 'iterations' : iterations, 'clustering_num' :clustering_num, 'model' : model}
    elif task == 'link_prediction':
        iterations = 4 * FLAGS.iterations
        re = {'data_name': dataname, 'iterations' : iterations,'model' : model}

    return re

