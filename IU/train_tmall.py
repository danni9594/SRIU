from __future__ import division
from __future__ import print_function
import os
import sys
import time
import numpy as np
from datetime import datetime
from engine import *
from model import *
from utils import *

np.random.seed(1234)
tf.set_random_seed(123)

data_df = pd.read_csv('../datasets/tmall_2014mth10_5neg.csv')
data_df.columns = ['userId', 'itemId', 'label', 'period']  # rename columns

num_users = data_df['userId'].max() + 1
num_items = data_df['itemId'].max() + 1

train_config = {'method': 'incremental_update',
                'dir_name': 'tmall_MF_IU_10.19-10.30',  # edit based on dataset, model, method, period range
                'pretrained_model': 'tmall_MF_pretrained_10.01-10.18_0.001',  # edit pretrained model
                'start_period': 20141019,  # overall train start period
                'end_period': 20141030,  # overall train end period
                'period_length': 1,
                'train_start_period': None,  # configure in the for loop
                'train_end_period': None,  # configure in the for loop
                'test_start_period': None,  # configure in the for loop
                'test_end_period': None,  # configure in the for loop
                'model_alias': None,  # configure in the for loop
                'restored_ckpt_mode': 'last',  # mode to search the checkpoint to restore, 'best auc', 'best gauc', 'last'
                'restored_ckpt': None,  # configure in the for loop

                'optimizer': 'adam',
                'lr': None,  # configure in the for loop
                'bs': 1024,
                'num_epochs': 1,
                'shuffle': True
                }

MF_hyperparams = {'num_users': num_users,
                  'num_items': num_items,
                  'user_embed_dim': 32,
                  'item_embed_dim': 32
                  }


def train():

    # create an engine instance
    engine = Engine(checkpoints_dir, sess, model)

    train_start_time = time.time()

    max_auc = 0

    for epoch_id in range(1, train_config['num_epochs'] + 1):

        avg_loss = engine.train_an_epoch(epoch_id, train_set, train_config)
        test_auc, test_gauc = engine.test(epoch_id, test_set, train_config)

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

        if test_auc > max_auc:
            max_auc = test_auc

    test_aucs.append(max_auc)

    engine.writer.close()
    print('')


# build model computation graph
model = MF(MF_hyperparams, train_config=train_config)

# create session
sess = tf.InteractiveSession()

# create saver
saver = tf.train.Saver(max_to_keep=200)

orig_dir_name = train_config['dir_name']

for lr in [1e-2]:

    print('')
    print('lr', lr)

    train_config['lr'] = lr

    train_config['dir_name'] = orig_dir_name + '_' + str(lr)
    print('model alias: ', train_config['dir_name'])

    test_aucs = []

    for period in range(train_config['start_period'], train_config['end_period'] + 1, train_config['period_length']):

        # configure train_start_period, train_end_period, test_start_period, test_end_period, model_alias, restored_ckpt
        train_config['train_start_period'] = period
        train_config['train_end_period'] = period + train_config['period_length'] - 1
        train_config['test_start_period'] = period + train_config['period_length']
        train_config['test_end_period'] = period + train_config['period_length'] + train_config['period_length'] - 1
        print('')
        print('train period range: {} - {}, test period range: {} - {}'.format(
            train_config['train_start_period'],
            train_config['train_end_period'],
            train_config['test_start_period'],
            train_config['test_end_period']))
        print('')

        train_config['model_alias'] = 'MF_IU_' + datetime.strptime(str(period), '%Y%m%d').strftime('%m.%d')

        if period == train_config['start_period']:
            search_alias = os.path.join('../pretrained/ckpts', train_config['pretrained_model'], 'Epoch*')
            train_config['restored_ckpt'] = search_ckpt(search_alias, mode='best auc')
            print('restored checkpoint: {}'.format(train_config['restored_ckpt']))
        else:
            prev_model_alias = 'MF_IU_' + datetime.strptime(str(period - train_config['period_length']), '%Y%m%d').strftime('%m.%d')
            if train_config['restored_ckpt_mode'] == 'last':
                search_alias = os.path.join('ckpts', train_config['dir_name'], prev_model_alias, 'Epoch{}*'.format(train_config['num_epochs']))
            else:
                search_alias = os.path.join('ckpts', train_config['dir_name'], prev_model_alias, 'Epoch*')
            train_config['restored_ckpt'] = search_ckpt(search_alias, mode=train_config['restored_ckpt_mode'])
            print('restored checkpoint: {}'.format(train_config['restored_ckpt']))

        # create train and test set
        sample_generator = SampleGenerator(data_df, train_config)
        train_set = sample_generator.train_set()
        test_set = sample_generator.test_set()

        # ckpt and config saving directory
        checkpoints_dir = os.path.join('ckpts', train_config['dir_name'], train_config['model_alias'])
        if not os.path.exists(checkpoints_dir):
            os.makedirs(checkpoints_dir)

        # write train_config to text file
        with open(os.path.join(checkpoints_dir, 'config.txt'), mode='w') as f:
            f.write('train_config: ' + str(train_config) + '\n')
            f.write('\n')
            f.write('MF_hyperparams: ' + str(MF_hyperparams) + '\n')

        saver.restore(sess, train_config['restored_ckpt'])
        train()

    average_auc = sum(test_aucs) / len(test_aucs)
    print('test aucs', test_aucs)
    print('average auc', average_auc)

    # write config to text file
    with open(os.path.join(checkpoints_dir, 'test_auc.txt'), mode='w') as f:
        f.write('test_aucs: ' + str(test_aucs) + '\n')
        f.write('average_auc: ' + str(average_auc) + '\n')
