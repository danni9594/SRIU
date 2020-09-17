import pandas as pd
import tensorflow as tf
import glob


class SampleGenerator:
    """
    generate train and test set based on specified period range
    return: df ['userId', 'itemId', 'label']
    """

    def __init__(self, data_df, config):

        self.data_df = data_df  # ['userId', 'itemId', 'label', 'period']
        self.config = config

    def train_set(self):
        train_set = self.data_df[(self.data_df['period'] >= self.config['train_start_period'])
                                 & (self.data_df['period'] <= self.config['train_end_period'])]
        train_set = train_set.drop(['period'], axis=1)
        return train_set

    def test_set(self):
        test_set = self.data_df[(self.data_df['period'] >= self.config['test_start_period'])
                                & (self.data_df['period'] <= self.config['test_end_period'])]
        test_set = test_set.drop(['period'], axis=1)
        return test_set


class DataLoader:
    """
    train set batch data loader
    return: [[users], [items], [labels]] in batch iterator
    """

    def __init__(self, data, batch_size):

        self.data = data  # df ['userId', 'itemId', 'label']
        self.batch_size = batch_size

        # number of batches in one epoch
        self.epoch_size = - (len(data) // -batch_size)

        # track current batch id
        self.id = 0

    def __iter__(self):
        return self

    def next(self):

        if self.id == self.epoch_size:
            raise StopIteration

        # retrieve train samples for current batch
        batch = self.data[self.id * self.batch_size: min((self.id + 1) * self.batch_size, len(self.data))]

        self.id += 1

        users = batch['userId'].tolist()
        items = batch['itemId'].tolist()
        labels = batch['label'].tolist()

        return self.id, [users, items, labels]


class WeightDataLoader:
    """
    train set batch data loader
    return: [[users], [items], [labels], [weights]] in batch iterator
    """

    def __init__(self, data, batch_size):

        self.data = data  # df ['userId', 'itemId', 'label']
        self.batch_size = batch_size

        # number of batches in one epoch
        self.epoch_size = - (len(data) // -batch_size)

        # track current batch id
        self.id = 0

    def __iter__(self):
        return self

    def next(self):

        if self.id == self.epoch_size:
            raise StopIteration

        # retrieve train samples for current batch
        batch = self.data[self.id * self.batch_size: min((self.id + 1) * self.batch_size, len(self.data))]

        self.id += 1

        users = batch['userId'].tolist()
        items = batch['itemId'].tolist()
        labels = batch['label'].tolist()
        weights = batch['weight'].tolist()

        return self.id, [users, items, labels, weights]


def variable_summaries(var, var_name):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope(var_name):
        mean = tf.reduce_mean(var)
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('mean', mean)
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)


def cal_roc_auc(scores, labels):

    arr = sorted(zip(scores, labels), key=lambda d: d[0], reverse=True)
    pos, neg = 0., 0.
    for record in arr:
        if record[1] == 1.:
            pos += 1
        else:
            neg += 1

    if pos == 0 or neg == 0:
        return None

    fp, tp = 0., 0.
    xy_arr = []
    for record in arr:
        if record[1] == 1.:
            tp += 1
        else:
            fp += 1
        xy_arr.append([fp/neg, tp/pos])

    auc = 0.
    prev_x = 0.
    prev_y = 0.
    for x, y in xy_arr:
        auc += ((x - prev_x) * (y + prev_y) / 2.)
        prev_x = x
        prev_y = y
    return auc


def cal_roc_gauc(users, scores, labels):
    # weighted sum of individual auc
    df = pd.DataFrame({'user': users,
                       'score': scores,
                       'label': labels})

    df_gb = df.groupby('user').agg(lambda x: x.tolist())

    auc_ls = []  # collect auc for all users
    user_imp_ls = []

    for row in df_gb.itertuples():
        auc = cal_roc_auc(row.score, row.label)
        if auc is None:
            pass
        else:
            auc_ls.append(auc)
            user_imp = len(row.label)
            user_imp_ls.append(user_imp)

    total_imp = sum(user_imp_ls)
    weighted_auc_ls = [auc * user_imp / total_imp for auc, user_imp in zip(auc_ls, user_imp_ls)]

    return sum(weighted_auc_ls)


def search_ckpt(search_alias, mode='best auc'):
    ckpt_ls = glob.glob(search_alias)

    if mode == 'last':
        ckpt = ckpt_ls[0]
    else:
        if mode == 'best gauc':
            metrics_ls = [int(ckpt.split('_')[-1].split('.')[-3]) for ckpt in ckpt_ls]  # gauc
        else:
            metrics_ls = [int(ckpt.split('_')[-2].split('.')[-1]) for ckpt in ckpt_ls]  # auc
        max_metrics_pos_ls = [i for i, x in enumerate(metrics_ls) if x == max(metrics_ls)]  # find all positions of the best ckpts
        ckpt = ckpt_ls[max(max_metrics_pos_ls)]  # get the full path of the last best ckpt

    ckpt = ckpt.split('.ckpt')[0]  # get the path before .ckpt
    ckpt = ckpt + '.ckpt'  # get the path with .ckpt
    return ckpt


def save_checkpoint(sess, path):
    saver = tf.train.Saver()
    saver.save(sess, save_path=path)


def restore_checkpoint(sess, path):
    saver = tf.train.Saver()
    saver.restore(sess, save_path=path)
