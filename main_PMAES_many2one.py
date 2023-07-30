import os
import time
import argparse
import random
import numpy as np
from torch.utils.data import DataLoader

from models.PMAES import EssayEncoder, Scorer, PromptMappingCL
from configs import Configs
from utils.read_data import read_pos_vocab, read_essays_single_score
from utils.general_utils import pad_hierarchical_text_sequences, get_single_scaled_down_score

import torch
from data_set import PMAESDataSet
from train.train_PMAES import TrainSingleOverallScoring


def seed_all(seed_value):
    """
    Setting the random seed across the pipeline
    """
    torch.manual_seed(seed_value) # cpu  vars
    random.seed(seed_value)
    np.random.seed(seed_value) # cpu vars
    os.environ['PYTHONHASHSEED'] = str(seed_value)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)  # gpu vars
        torch.backends.cudnn.deterministic = True  # needed
        torch.backends.cudnn.benchmark = False


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="PAES_attributes models")
    parser.add_argument('--seed', type=int, default=12, help='set random seed')
    parser.add_argument('--target_prompt_id', type=int, default=1, help='set random seed')
    parser.add_argument('--attribute_name', type=str, default='score', help='set random seed')
    parser.add_argument('--source2target', type=str, default='many2one', help='Setting of source-target pair')
    parser.add_argument('--embedding_dim', type=int, default=50, help='Only useful when embedding is randomly initialised')
    parser.add_argument('--num_epochs', type=int, default=50, help='number of epochs for training')
    parser.add_argument('--filter_num', type=int, default=100, help='Num of filters in conv layer')
    parser.add_argument('--kernel_size', type=int, default=3, help='filter length in 1st conv layer')
    parser.add_argument('--rnn_type', type=str, default='LSTM', help='Recurrent type')
    parser.add_argument('--lstm_units', type=int, default=50, help='Num of hidden units in recurrent layer')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Initial learning rate')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate for layers')
    parser.add_argument('--device', type=str, help='cpu or gpu', default='cuda')

    args = parser.parse_args()
    test_prompt_id = args.target_prompt_id
    seed = args.seed

    seed_all(seed)

    print("Test prompt id is {} of type {}".format(test_prompt_id, type(test_prompt_id)))
    print("Seed: {}".format(seed))
    configs = Configs()

    data_path = configs.DATA_PATH
    train_path = data_path + str(test_prompt_id) + '/train.pk'
    dev_path = data_path + str(test_prompt_id) + '/dev.pk'
    test_path = data_path + str(test_prompt_id) + '/test.pk'
    features_path = configs.FEATURES_PATH
    readability_path = configs.READABILITY_PATH

    read_configs = {
        'train_path': train_path,
        'dev_path': dev_path,
        'test_path': test_path,
        'features_path': features_path,
        'readability_path': readability_path,
    }

    pos_vocab = read_pos_vocab(read_configs)
    train_data, valid_data, test_data = read_essays_single_score(read_configs, pos_vocab, args.attribute_name)

    max_sent_len = max(train_data['max_sent_len'], valid_data['max_sent_len'], test_data['max_sent_len'])
    max_sent_num = max(train_data['max_sent_num'], valid_data['max_sent_num'], test_data['max_sent_num'])
    print('max sent length: {}'.format(max_sent_len))
    print('max sent num: {}'.format(max_sent_num))
    train_data['score_scaled'] = get_single_scaled_down_score(train_data['score'], train_data['prompt_ids'], args.attribute_name)
    valid_data['score_scaled'] = get_single_scaled_down_score(valid_data['score'], valid_data['prompt_ids'], args.attribute_name)
    test_data['score_scaled'] = get_single_scaled_down_score(test_data['score'], test_data['prompt_ids'], args.attribute_name)

    train_prompt_ids = train_data['prompt_ids']
    dev_prompt_ids = valid_data['prompt_ids']
    test_prompt_ids = test_data['prompt_ids']
    train_essay_pos = pad_hierarchical_text_sequences(train_data['essay_pos'], max_sent_num, max_sent_len)
    valid_essay_pos = pad_hierarchical_text_sequences(valid_data['essay_pos'], max_sent_num, max_sent_len)
    test_essay_pos = pad_hierarchical_text_sequences(test_data['essay_pos'], max_sent_num, max_sent_len)

    train_essay_pos = train_essay_pos.reshape((train_essay_pos.shape[0], train_essay_pos.shape[1] * train_essay_pos.shape[2]))
    valid_essay_pos = valid_essay_pos.reshape((valid_essay_pos.shape[0], valid_essay_pos.shape[1] * valid_essay_pos.shape[2]))
    test_essay_pos = test_essay_pos.reshape((test_essay_pos.shape[0], test_essay_pos.shape[1] * test_essay_pos.shape[2]))

    train_score = np.array(train_data['score_scaled'])
    valid_score = np.array(valid_data['score_scaled'])
    test_score = np.array(test_data['score_scaled'])

    train_linguistic = np.array(train_data['linguistic'])
    valid_linguistic = np.array(valid_data['linguistic'])
    test_linguistic = np.array(test_data['linguistic'])

    train_readability = np.array(train_data['readability'])
    valid_readability = np.array(valid_data['readability'])
    test_readability = np.array(test_data['readability'])

    tr_s_num, tr_t_num = len(train_essay_pos), len(test_essay_pos)
    batch_num = 2
    s_batch_size = int(tr_s_num / batch_num)
    t_batch_size = int(tr_t_num / batch_num)
    s_batch_num = int(tr_s_num / s_batch_size)
    t_batch_num = int(tr_t_num / t_batch_size)
    while t_batch_num > s_batch_num:
        s_batch_size -= 1
        s_batch_num = int(tr_s_num / s_batch_size)

    tr_s_loader = DataLoader(PMAESDataSet(train_prompt_ids, train_essay_pos, train_linguistic, train_readability, train_score), batch_size=s_batch_size)
    va_s_loader = DataLoader(PMAESDataSet(dev_prompt_ids, valid_essay_pos, valid_linguistic, valid_readability, valid_score), batch_size=s_batch_size)
    te_t_loader = DataLoader(PMAESDataSet(test_prompt_ids, test_essay_pos, test_linguistic, test_readability, test_score), batch_size=t_batch_size)

    essay_encoder = EssayEncoder(args, max_num=max_sent_num, max_len=max_sent_len, embed_dim=args.embedding_dim, pos_vocab=pos_vocab).to(args.device)
    scorer = Scorer(args).to(args.device)
    pm_cl = PromptMappingCL(args, tr_s_num, tr_t_num).to(args.device)
    optims = torch.optim.Adam([{'params': essay_encoder.parameters()}, {'params': scorer.parameters()}, {'params': pm_cl.parameters()}], lr=args.learning_rate)
    tr_log = {
        'Epoch_best_dev_qwk': [0, 0, 0],
        'Best_dev_qwk': [0, 0],
    }
    epochs = 50
    for e_index in range(1, epochs+1):

        TrainSingleOverallScoring(args,
                                  essay_encoder, scorer, pm_cl, optims,
                                  tr_s_loader, va_s_loader, te_t_loader,
                                  args.target_prompt_id, e_index,
                                  tr_log, args.attribute_name)
    file_time = time.strftime("%Y-%m-%d", time.localtime())
    result_file = 'result/many2one/PMAES-{}.txt'.format(seed)
    with open(result_file, 'a', encoding='utf-8') as f:
        f.write(file_time + 'TargetPrompt: {} Trait: {}'.format(args.target_id, args.attribute_name) + '\n')
        for key, value in tr_log.items():
            f.write(key + ':' + str(value) + '\n')



