from utils import *


class Engine(object):
    """
    Training epoch and test
    """

    def __init__(self, checkpoints_dir, sess, model):

        self.sess = sess
        self.model = model
        self.writer = tf.summary.FileWriter(checkpoints_dir, sess.graph)

    def add_summary(self, tags, values, epoch_id):
        # for multiple epochs
        summary = tf.Summary()
        for i in range(len(tags)):
            summary.value.add(tag=tags[i], simple_value=values[i])
        self.writer.add_summary(summary, epoch_id)
        self.writer.flush()

    def train_an_epoch(self, epoch_id, train_set, train_config):

        print('Epoch {} Start!'.format(epoch_id))

        if train_config['shuffle']:
            train_set = train_set.sample(frac=1)

        train_data_loader = DataLoader(train_set, train_config['bs'])
        loss_sum = 0

        train_batch_id = 0
        for train_batch_id, train_batch in train_data_loader:

            loss = self.model.train(self.sess, train_batch)  # sess.run

            if (train_batch_id - 1) % 500 == 0:

                print('[Epoch {} Batch {}] loss {:.4f}'.format(
                    epoch_id,
                    train_batch_id,
                    loss))

                # create batch output summary for last train_batch in epoch
                batch_summary = self.model.create_batch_summary(self.sess, train_batch)
                self.writer.add_summary(batch_summary, (epoch_id - 1) * train_data_loader.epoch_size + train_batch_id)
                self.writer.flush()

            loss_sum += loss

        # epoch done, compute average loss
        avg_loss = loss_sum / train_batch_id

        self.add_summary(tags=['average_loss'], values=[avg_loss], epoch_id=epoch_id)

        return avg_loss

    def test(self, epoch_id, test_set, train_config):

        test_data_loader = DataLoader(test_set, train_config['bs'])  # load batch test set

        users, scores, labels = [], [], []
        for test_batch_id, test_batch in test_data_loader:
            users.extend(test_batch[0])
            scores.extend(self.model.inference(self.sess, test_batch).tolist())  # sees.run
            labels.extend(test_batch[2])

        test_auc = cal_roc_auc(scores, labels)
        test_gauc = cal_roc_gauc(users, scores, labels)

        self.add_summary(tags=['test_auc', 'test_gauc'], values=[test_auc, test_gauc], epoch_id=epoch_id)

        return test_auc, test_gauc
