from __future__ import division
from __future__ import print_function
import os
# Train on CPU (hide GPU) due to memory constraints
os.environ['CUDA_VISIBLE_DEVICES'] = ""

import tensorflow as tf
import settings
from constructor import get_placeholder, get_model, format_data, get_optimizer, update
from metrics import linkpred_metrics
# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS

class Link_pred_Runner():
    def __init__(self, settings):
        self.data_name = settings['data_name']
        self.iteration = settings['iterations']
        self.model = settings['model']

    def erun(self):
        model_str = self.model
        # formatted data
        feas = format_data(self.data_name)

        # Define placeholders
        placeholders = get_placeholder(feas['adj'])

        # construct model
        d_real, discriminator, ae_model = get_model(model_str, placeholders, feas['num_features'], feas['num_nodes'], feas['features_nonzero'])

        # Optimizer
        opt = get_optimizer(model_str, ae_model, discriminator, placeholders, feas['pos_weight'], feas['norm'], d_real, feas['num_nodes'])

        # Initialize session
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        val_roc_score = []

        # Train model
        for epoch in range(self.iteration):

            emb, avg_cost = update(ae_model, opt, sess, feas['adj_norm'], feas['adj_label'], feas['features'], placeholders, feas['adj'])

            lm_train = linkpred_metrics(feas['val_edges'], feas['val_edges_false'])
            roc_curr, ap_curr, _ = lm_train.get_roc_score(emb, feas)
            val_roc_score.append(roc_curr)

            print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(avg_cost), "val_roc=", "{:.5f}".format(val_roc_score[-1]), "val_ap=", "{:.5f}".format(ap_curr))

            if (epoch+1) % 10 == 0:
                lm_test = linkpred_metrics(feas['test_edges'], feas['test_edges_false'])
                roc_score, ap_score,_ = lm_test.get_roc_score(emb, feas)
                print('Test ROC score: ' + str(roc_score))
                print('Test AP score: ' + str(ap_score))