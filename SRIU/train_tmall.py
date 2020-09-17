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

train_config = {'method': 'sample_reweight',
                'dir_name': 'tmall_MF_SRIU_10.19-10.30',  # edit based on dataset, model, method, period range
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
                'shuffle': True,

                'score_shrinkage': None,  # configure in the for loop
                }

MF_hyperparams = {'num_users': num_users,
                  'num_items': num_items,
                  'user_embed_dim': 32,
                  'item_embed_dim': 32
                  }


def add_weights_to_train_set():

    reweight_start_time = time.time()

    # compute logits using initialized model
    logits = []
    train_data_loader = DataLoader(train_set, 1024)  # load batch train set, bs = 1024
    for train_batch_id, train_batch in train_data_loader:
        logits.extend(model.compute_logits(sess, train_batch).tolist())  # sees.run

    train_set_w_weight = train_set
    train_set_w_weight['logits'] = logits

    # compute neg_logits_mean, neg_logits_std, pos_logits_mean, pos_logits_std for each individual user
    neg_mean_by_user = train_set_w_weight[train_set_w_weight['label'] == 0].groupby('userId')[
        'logits'].mean().reset_index().rename(columns={'logits': 'neg_logits_mean'})
    neg_std_by_user = train_set_w_weight[train_set_w_weight['label'] == 0].groupby('userId')[
        'logits'].std().reset_index().rename(columns={'logits': 'neg_logits_std'})
    pos_mean_by_user = train_set_w_weight[train_set_w_weight['label'] == 1].groupby('userId')[
        'logits'].mean().reset_index().rename(columns={'logits': 'pos_logits_mean'})
    pos_std_by_user = train_set_w_weight[train_set_w_weight['label'] == 1].groupby('userId')[
        'logits'].std().reset_index().rename(columns={'logits': 'pos_logits_std'})
    train_set_w_weight = train_set_w_weight.merge(neg_mean_by_user, on='userId', how='left').merge(neg_std_by_user, on='userId', how='left').merge(
        pos_mean_by_user, on='userId', how='left').merge(pos_std_by_user, on='userId', how='left')

    # if only have 1 (neg or pos) sample, will get na std
    train_set_w_weight = train_set_w_weight.fillna(1)

    # if all (neg or pos) samples have the same logits, will get 0 std
    train_set_w_weight['pos_logits_std'].replace(0, 1, inplace=True)
    train_set_w_weight['neg_logits_std'].replace(0, 1, inplace=True)

    # standardization
    train_set_w_weight['score'] = train_set_w_weight.apply(
        lambda x: -(x['logits'] - x['neg_logits_mean']) / x['neg_logits_std'] * train_config['score_shrinkage'] if x['label'] == 0
        else (x['logits'] - x['pos_logits_mean']) / x['pos_logits_std'] * train_config['score_shrinkage'], axis=1)

    # normalization
    expsum_by_user = train_set_w_weight.groupby('userId')['score'].apply(lambda x: np.sum(np.exp(x))).reset_index().rename(columns={'score': 'score_expsum'})
    count_per_user = train_set_w_weight.groupby('userId')['score'].count().reset_index().rename(columns={'score': 'count'})
    train_set_w_weight = train_set_w_weight.merge(expsum_by_user, on='userId', how='left').merge(count_per_user, on='userId', how='left')
    train_set_w_weight['weight'] = train_set_w_weight.apply(lambda x: np.exp(x['score']) / x['score_expsum'] * x['count'], axis=1)

    # save for visualization
    save_path = os.path.join(checkpoints_dir, 'weight.csv')
    with open(save_path, mode='w') as wf:
        train_set_w_weight.to_csv(wf, index=False)

    train_set_w_weight = train_set_w_weight.drop(['logits', 'neg_logits_mean', 'neg_logits_std', 'pos_logits_mean', 'pos_logits_std', 'score', 'score_expsum', 'count'], axis=1)

    print('Done adding weights! time elapsed: {}'.format(time.strftime('%H:%M:%S', time.gmtime(time.time() - reweight_start_time))))

    return train_set_w_weight


def train():

    # create an engine instance
    engine = Engine(checkpoints_dir, sess, model)

    train_start_time = time.time()

    max_auc = 0

    for epoch_id in range(1, train_config['num_epochs'] + 1):

        avg_loss = engine.train_an_epoch(epoch_id, train_set_with_weight, train_config)
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

for shrinkage in [0.5]:

    for lr in [1e-2]:

        print('')
        print('shrinkage', shrinkage, 'lr', lr)

        train_config['score_shrinkage'] = shrinkage
        train_config['lr'] = lr

        train_config['dir_name'] = orig_dir_name + '_' + str(shrinkage) + '_' + str(lr)
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

            train_config['model_alias'] = 'MF_SRIU_' + datetime.strptime(str(period), '%Y%m%d').strftime('%m.%d')

            if period == train_config['start_period']:
                search_alias = os.path.join('../pretrained/ckpts', train_config['pretrained_model'], 'Epoch*')
                train_config['restored_ckpt'] = search_ckpt(search_alias, mode='best auc')
                print('restored checkpoint: {}'.format(train_config['restored_ckpt']))
            else:
                prev_model_alias = 'MF_SRIU_' + datetime.strptime(str(period - train_config['period_length']), '%Y%m%d').strftime('%m.%d')
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
            train_set_with_weight = add_weights_to_train_set()
            train()

        average_auc = sum(test_aucs) / len(test_aucs)
        print('test aucs', test_aucs)
        print('average auc', average_auc)

        # write config to text file
        with open(os.path.join(checkpoints_dir, 'test_auc.txt'), mode='w') as f:
            f.write('test_aucs: ' + str(test_aucs) + '\n')
            f.write('average_auc: ' + str(average_auc) + '\n')
