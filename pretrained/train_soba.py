from __future__ import division
from __future__ import print_function
import os
import sys
import time
import numpy as np
from engine import *
from model import *
from utils import *

np.random.seed(1234)
tf.set_random_seed(123)

data_df = pd.read_csv('../datasets/soba_2014mth09-12_5neg.csv')
data_df.columns = ['userId', 'itemId', 'label', 'period']  # rename columns

num_users = data_df['userId'].max() + 1
num_items = data_df['itemId'].max() + 1

pretrained_config = {'method': 'pretrained',
                     'train_start_period': 1,
                     'train_end_period': 18,
                     'test_start_period': 19,
                     'test_end_period': 19,
                     'model_alias': 'soba_MF_pretrained_1-18',  # edit based on dataset, model, method, period range

                     'optimizer': 'adam',
                     'lr': None,  # configure in the for loo
                     'bs': 256,
                     'num_epochs': 10,
                     'shuffle': True,
                     }

MF_hyperparams = {'num_users': num_users,
                  'num_items': num_items,
                  'user_embed_dim': 64,
                  'item_embed_dim': 64
                  }


def train():

    # create an engine instance
    engine = Engine(checkpoints_dir, sess, model)

    train_start_time = time.time()

    for epoch_id in range(1, pretrained_config['num_epochs'] + 1):

        avg_loss = engine.train_an_epoch(epoch_id, train_set, pretrained_config)
        test_auc, test_gauc = engine.test(epoch_id, test_set, pretrained_config)

        print('Epoch {} Done! time elapsed: {}, average loss {:.4f}, test auc {:.4f}, test gauc {:.4f}'.format(
            epoch_id,
            time.strftime('%H:%M:%S', time.gmtime(time.time() - train_start_time)),
            avg_loss,
            test_auc,
            test_gauc))
        sys.stdout.flush()

        # save checkpoint
        checkpoint_alias = 'Epoch{}_TestAUC{:.4f}_TestGAUC{:.4f}.ckpt'.format(
            epoch_id,
            test_auc,
            test_gauc)
        checkpoint_path = os.path.join(checkpoints_dir, checkpoint_alias)
        saver.save(sess, checkpoint_path)

    engine.writer.close()
    print('')


# build model computation graph
model = MF(MF_hyperparams, train_config=pretrained_config)

# create session
sess = tf.InteractiveSession()

# create saver
saver = tf.train.Saver(max_to_keep=30)

orig_dir_name = pretrained_config['model_alias']

for lr in [1e-3]:

    print('')
    print('lr', lr)

    pretrained_config['lr'] = lr

    pretrained_config['model_alias'] = orig_dir_name + '_' + str(lr)
    print('model alias: ', pretrained_config['model_alias'])

    # create train and test set
    sample_generator = SampleGenerator(data_df, pretrained_config)
    train_set = sample_generator.train_set()
    test_set = sample_generator.test_set()

    # ckpt and config saving directory
    checkpoints_dir = os.path.join('ckpts', pretrained_config['model_alias'])
    if not os.path.exists(checkpoints_dir):
        os.makedirs(checkpoints_dir)

    # write pretrained_config to text file
    with open(os.path.join(checkpoints_dir, 'config.txt'), mode='w') as f:
        f.write('pretrained_config: ' + str(pretrained_config) + '\n')
        f.write('\n')
        f.write('MF_hyperparams: ' + str(MF_hyperparams) + '\n')

    sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
    train()
