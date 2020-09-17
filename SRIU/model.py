import tensorflow as tf
from utils import variable_summaries


class MF(object):

    def __init__(self, hyperparams, train_config):

        self.train_config = train_config

        # create placeholder
        self.u = tf.placeholder(tf.int32, [None])  # [B]
        self.i = tf.placeholder(tf.int32, [None])  # [B]
        self.y = tf.placeholder(tf.float32, [None])  # [B]
        self.w = tf.placeholder(tf.float32, [None])  # [B]
        self.lr = tf.placeholder(tf.float32, [], name='learning_rate')

        # -- create embed begin ----
        user_emb_w = tf.get_variable("user_emb_w", [hyperparams['num_users'], hyperparams['user_embed_dim']])
        item_emb_w = tf.get_variable("item_emb_w", [hyperparams['num_items'], hyperparams['item_embed_dim']])
        user_b = tf.get_variable("user_b", [hyperparams['num_users']], initializer=tf.constant_initializer(0.0))
        item_b = tf.get_variable("item_b", [hyperparams['num_items']], initializer=tf.constant_initializer(0.0))
        # -- create embed end ----

        # -- embed begin -------
        u_emb = tf.nn.embedding_lookup(user_emb_w, self.u)
        i_emb = tf.nn.embedding_lookup(item_emb_w, self.i)
        u_b = tf.gather(user_b, self.u)  # [B]
        i_b = tf.gather(item_b, self.i)  # [B]
        # -- embed end -------

        interaction = tf.reduce_sum(u_emb * i_emb, axis=-1)  # [B]
        self.logits = interaction + u_b + i_b  # [B]
        self.scores = tf.nn.sigmoid(self.logits)  # scores is logits into sigmoid, for inference

        variable_summaries(self.logits, 'logits')
        variable_summaries(self.scores, 'scores')

        # return same dimension as input tensors, let x = logits, z = labels, z * -log(sigmoid(x)) + (1 - z) * -log(1 - sigmoid(x))
        self.losses = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits, labels=self.y)
        variable_summaries(self.losses, 'loss')

        self.loss = tf.reduce_mean(self.losses * self.w)  # for training loss

        # global update step variable
        self.global_step = tf.Variable(0, trainable=False, name='global_step')

        # optimizer
        if train_config['optimizer'] == 'adam':
            optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        elif train_config['optimizer'] == 'rmsprop':
            optimizer = tf.train.RMSPropOptimizer(learning_rate=self.lr)
        else:
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.lr)

        # compute gradients and different update step
        trainable_params = tf.trainable_variables()
        grads = tf.gradients(self.loss, trainable_params)  # return a list of gradients (A list of `sum(dy/dx)` for each x in `xs`)
        clip_grads, _ = tf.clip_by_global_norm(grads, 5)
        clip_grads_tuples = zip(clip_grads, trainable_params)
        self.train_op = optimizer.apply_gradients(clip_grads_tuples, global_step=self.global_step)

        self.merged = tf.summary.merge_all()

    def train(self, sess, batch):
        loss, _ = sess.run([self.loss, self.train_op], feed_dict={
            self.u: batch[0],
            self.i: batch[1],
            self.y: batch[2],
            self.w: batch[3],
            self.lr: self.train_config['lr']
        })
        return loss

    def inference(self, sess, batch):
        scores = sess.run(self.scores, feed_dict={
            self.u: batch[0],
            self.i: batch[1]
        })
        return scores

    def compute_logits(self, sess, batch):
        logits = sess.run(self.logits, feed_dict={
            self.u: batch[0],
            self.i: batch[1]
        })
        return logits

    def create_batch_summary(self, sess, batch):
        batch_summary = sess.run(self.merged, feed_dict={
            self.u: batch[0],
            self.i: batch[1],
            self.y: batch[2],
            self.w: batch[3]
        })
        return batch_summary
