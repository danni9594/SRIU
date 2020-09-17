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

random.seed(123)
np.random.seed(1234)
tf.set_random_seed(123)

data_df = pd.read_csv('../datasets/tmall_2014mth10_5neg.csv')
data_df.columns = ['userId', 'itemId', 'label', 'period']  # rename columns

num_users = data_df['userId'].max() + 1
num_items = data_df['itemId'].max() + 1

train_config = {'method': 'spmf',
                'dir_name': 'tmall_MF_SPMF_10.19-10.30',  # edit based on dataset, model, method, period range
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

                'strategy': 2,  # two different sampling strategies
                'frac_of_history_D': None,  # less than or equal to 1
                'res_cur_ratio': None,  # the ratio of reservoir sample to current set, only for strategy 2
                }

MF_hyperparams = {'num_users': num_users,
                  'num_items': num_items,
                  'user_embed_dim': 32,
                  'item_embed_dim': 32
                  }


def compute_prob_and_gen_set_and_update_reservoir():

    """
    this strategy follows exactly the method from the paper "Streaming ranking based recommender systems"
    train_set = random samples of (current_set + reservoir)
    """
    compute_prob_start_time = time.time()

    # compute prob
    pos_train_data_loader = DataLoader(pos_train_set, int(len(pos_train_set) / 100))

    scores = []
    for pos_train_batch_id, pos_train_batch in pos_train_data_loader:
        batch_scores = model.inference(sess, pos_train_batch)  # sess.run
        scores.extend(batch_scores)

    ordered_pos_train_set = pos_train_set
    ordered_pos_train_set['score'] = scores
    ordered_pos_train_set = ordered_pos_train_set.sort_values(['score'], ascending=False).reset_index(drop=True)
    ordered_pos_train_set['rank'] = np.arange(len(ordered_pos_train_set))
    total_num = len(pos_train_set)
    ordered_pos_train_set['weight'] = ordered_pos_train_set['rank'].apply(lambda x: np.exp(x / total_num))
    total_weights = ordered_pos_train_set['weight'].sum()
    ordered_pos_train_set['prob'] = ordered_pos_train_set['weight'].apply(lambda x: x / total_weights)
    ordered_pos_train_set = ordered_pos_train_set.drop(['score', 'rank', 'weight'], axis=1)

    # generate train set
    sampled_pos_train_set = ordered_pos_train_set.sample(n=len(pos_current_set), replace=False, weights='prob')
    users, neg_items, labels = [], [], []
    pos_train_set_gb = pos_train_set.groupby(['userId'])
    for row in sampled_pos_train_set.itertuples():
        user_df = pos_train_set_gb.get_group(row.userId)
        pos_ls = user_df['itemId'].unique().tolist()
        neg_ls = gen_neg(MF_hyperparams['num_items'], pos_ls, num_neg=5)
        for neg_item in neg_ls:
            users.append(row.userId)
            neg_items.append(neg_item)
            labels.append(0)
    sampled_neg_train_set = pd.DataFrame({'userId': users,
                                          'itemId': neg_items,
                                          'label': labels})
    sampled_pos_train_set = sampled_pos_train_set.drop(['period', 'prob'], axis=1)
    sampled_train_set = pd.concat([sampled_pos_train_set, sampled_neg_train_set], ignore_index=False, sort=True)

    # update reservoir
    t = len(data_df[(data_df['period'] < period) & (data_df['label'] == 1)])
    probs_to_res = len(reservoir) / (t + np.arange(len(pos_current_set)) + 1)
    random_probs = np.random.rand(len(pos_current_set))
    selected_pos_current_set = pos_current_set[probs_to_res > random_probs]
    print('selected_pos_current_set size', len(selected_pos_current_set))
    print('num_in_res', len(reservoir))
    num_left_in_res = len(reservoir) - len(selected_pos_current_set)
    print('num_left_in_res', num_left_in_res)
    updated_reservoir = pd.concat([reservoir.sample(n=num_left_in_res), selected_pos_current_set], ignore_index=False, sort=True)
    print('num_in_updated_res', len(updated_reservoir))

    print('compute prob and generate train set and update reservoir time elapsed: {}'.format(
        time.strftime('%H:%M:%S', time.gmtime(time.time() - compute_prob_start_time))))

    return sampled_train_set, updated_reservoir


def compute_prob_and_gen_set_and_update_reservoir2():
    """
    this strategy modify slightly the method from paper "Streaming ranking based recommender systems"
    train_set = current_set + random samples of reservoir
    """
    compute_prob_start_time = time.time()

    # compute prob
    reservoir_data_loader = DataLoader(reservoir, int(len(reservoir) / 100))

    scores = []
    for reservoir_batch_id, reservoir_batch in reservoir_data_loader:
        batch_scores = model.inference(sess, reservoir_batch)  # sess.run
        scores.extend(batch_scores)

    ordered_reservoir = reservoir
    ordered_reservoir['score'] = scores
    ordered_reservoir = ordered_reservoir.sort_values(['score'], ascending=False).reset_index(drop=True)
    ordered_reservoir['rank'] = np.arange(len(ordered_reservoir))
    total_num = len(reservoir)
    ordered_reservoir['weight'] = ordered_reservoir['rank'].apply(lambda x: np.exp(x / total_num))
    total_weights = ordered_reservoir['weight'].sum()
    ordered_reservoir['prob'] = ordered_reservoir['weight'].apply(lambda x: x / total_weights)
    ordered_reservoir = ordered_reservoir.drop(['score', 'rank', 'weight'], axis=1)

    # generate train set
    sampled_reservoir = ordered_reservoir.sample(n=int(len(pos_current_set) * train_config['res_cur_ratio']), replace=False, weights='prob')
    users, neg_items, labels = [], [], []
    reservoir_gb = reservoir.groupby(['userId'])
    for row in sampled_reservoir.itertuples():
        user_df = reservoir_gb.get_group(row.userId)
        pos_ls = user_df['itemId'].unique().tolist()
        neg_ls = gen_neg(MF_hyperparams['num_items'], pos_ls, num_neg=5)
        for neg_item in neg_ls:
            users.append(row.userId)
            neg_items.append(neg_item)
            labels.append(0)
    sampled_neg_reservoir = pd.DataFrame({'userId': users,
                                          'itemId': neg_items,
                                          'label': labels})
    sampled_reservoir = sampled_reservoir.drop(['period', 'prob'], axis=1)
    sampled_reservoir = pd.concat([sampled_reservoir, sampled_neg_reservoir], ignore_index=False, sort=True)
    print('sampled_reservoir size', len(sampled_reservoir))
    sampled_train_set = pd.concat([sampled_reservoir, current_set], ignore_index=False, sort=True)
    print('sampled_train_set size', len(sampled_train_set))

    # update reservoir
    t = len(data_df[(data_df['period'] < period) & (data_df['label'] == 1)])
    probs_to_res = len(reservoir) / (t + np.arange(len(pos_current_set)) + 1)
    random_probs = np.random.rand(len(pos_current_set))
    selected_pos_current_set = pos_current_set[probs_to_res > random_probs]
    print('selected_pos_current_set size', len(selected_pos_current_set))
    print('num_in_res', len(reservoir))
    num_left_in_res = len(reservoir) - len(selected_pos_current_set)
    print('num_left_in_res', num_left_in_res)
    updated_reservoir = pd.concat([reservoir.sample(n=num_left_in_res), selected_pos_current_set], ignore_index=False, sort=True)
    print('num_in_updated_res', len(updated_reservoir))

    print('compute prob and generate train set and update reservoir time elapsed: {}'.format(
        time.strftime('%H:%M:%S', time.gmtime(time.time() - compute_prob_start_time))))

    return sampled_train_set, updated_reservoir


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
saver = tf.train.Saver(max_to_keep=200)  # make it large enough to accommodate n epochs in k training periods >= n x k

orig_dir_name = train_config['dir_name']

for frac in [0.2]:

    for ratio in [0.1]:

        for lr in [1e-2]:

            for dropout_prob in [1]:

                print('')
                # print('frac_of_history_D', frac, 'lr', lr, 'dropout', dropout_prob)
                print('frac_of_history_D', frac, 'res_cur_ratio', ratio, 'lr', lr, 'dropout', dropout_prob)

                train_config['frac_of_history_D'] = frac
                train_config['res_cur_ratio'] = ratio
                train_config['lr'] = lr
                train_config['dropout_prob'] = dropout_prob

                # IL_config['dir_name'] = orig_dir_name + '_' + str(frac) + '_' + str(lr) + '_' + str(dropout_prob)
                train_config['dir_name'] = orig_dir_name + '_' + str(frac) + '_' + str(ratio) + '_' + str(lr) + '_' + str(dropout_prob)
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

                    train_config['model_alias'] = 'MF_SPMF_' + datetime.strptime(str(period), '%Y%m%d').strftime('%m.%d')

                    if period == train_config['start_period']:
                        search_alias = os.path.join('../pretrained/ckpts', train_config['pretrained_model'], 'Epoch*')
                        train_config['restored_ckpt'] = search_ckpt(search_alias, mode='best auc')
                        print('restored checkpoint: {}'.format(train_config['restored_ckpt']))
                    else:
                        prev_model_alias = 'MF_SPMF_' + datetime.strptime(str(period - train_config['period_length']), '%Y%m%d').strftime('%m.%d')
                        if train_config['restored_ckpt_mode'] == 'last':
                            search_alias = os.path.join('ckpts', train_config['dir_name'], prev_model_alias, 'Epoch{}*'.format(train_config['num_epochs']))
                        else:
                            search_alias = os.path.join('ckpts', train_config['dir_name'], prev_model_alias, 'Epoch*')
                        train_config['restored_ckpt'] = search_ckpt(search_alias, mode=train_config['restored_ckpt_mode'])
                        print('restored checkpoint: {}'.format(train_config['restored_ckpt']))

                    # create train, test set
                    sample_generator = SampleGenerator(data_df, train_config)
                    test_set = sample_generator.test_set()  # ['userId', 'itemId', 'label']
                    current_set = sample_generator.current_set()  # ['userId', 'itemId', 'label']
                    pos_current_set = sample_generator.pos_current_set()  # ['userId', 'itemId', 'label' == 1, 'period']
                    if period == train_config['start_period']:
                        reservoir = sample_generator.init_reservoir()  # ['userId', 'itemId', 'label' == 1, 'period']
                    pos_train_set = pd.concat([reservoir, pos_current_set], ignore_index=False, sort=True)  # combine R and W

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
                    if train_config['strategy'] == 2:
                        train_set, reservoir = compute_prob_and_gen_set_and_update_reservoir2()
                    else:
                        train_set, reservoir = compute_prob_and_gen_set_and_update_reservoir()
                    train()

                average_auc = sum(test_aucs) / len(test_aucs)
                print('test aucs', test_aucs)
                print('average auc', average_auc)

                # write config to text file
                with open(os.path.join(checkpoints_dir, 'test_auc.txt'), mode='w') as f:
                    f.write('test_aucs: ' + str(test_aucs) + '\n')
                    f.write('average_auc: ' + str(average_auc) + '\n')
