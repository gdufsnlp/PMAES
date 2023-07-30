import logging
import sys
import numpy as np
from tqdm import tqdm


score_range = {
    1: [2, 12],
    2: [1, 6],
    3: [0, 3],
    4: [0, 3],
    5: [0, 4],
    6: [0, 4],
    7: [0, 30],
    8: [0, 60]
}


def get_min_max_scores():
    return {
        1: {'score': (2, 12), 'content': (1, 6), 'organization': (1, 6), 'word_choice': (1, 6),
            'sentence_fluency': (1, 6), 'conventions': (1, 6)},
        2: {'score': (1, 6), 'content': (1, 6), 'organization': (1, 6), 'word_choice': (1, 6),
            'sentence_fluency': (1, 6), 'conventions': (1, 6)},
        3: {'score': (0, 3), 'content': (0, 3), 'prompt_adherence': (0, 3), 'language': (0, 3), 'narrativity': (0, 3)},
        4: {'score': (0, 3), 'content': (0, 3), 'prompt_adherence': (0, 3), 'language': (0, 3), 'narrativity': (0, 3)},
        5: {'score': (0, 4), 'content': (0, 4), 'prompt_adherence': (0, 4), 'language': (0, 4), 'narrativity': (0, 4)},
        6: {'score': (0, 4), 'content': (0, 4), 'prompt_adherence': (0, 4), 'language': (0, 4), 'narrativity': (0, 4)},
        7: {'score': (0, 30), 'content': (0, 6), 'organization': (0, 6), 'conventions': (0, 6)},
        8: {'score': (0, 60), 'content': (2, 12), 'organization': (2, 12), 'word_choice': (2, 12),
            'sentence_fluency': (2, 12), 'conventions': (2, 12)}}


def get_score_vector_positions():
    return {
        'score': 0,
        'content': 1,
        'organization': 2,
        'word_choice': 3,
        'sentence_fluency': 4,
        'conventions': 5,
        'prompt_adherence': 6,
        'language': 7,
        'narrativity': 8,
        # 'style': 9,
        # 'voice': 10
    }


def get_logger(name, level=logging.INFO, handler=sys.stdout, formatter='%(name)s - %(levelname)s - %(message)s'):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(formatter)
    stream_handler = logging.StreamHandler(handler)
    stream_handler.setLevel(level)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger


def load_word_embedding_dict(embedding_path):
    embedding_dim = 50
    embedding_dict = {}

    with open(embedding_path, 'r', encoding='utf-8')as f:
        for line in tqdm(f.readlines(), desc='Load glove embedding'):
            line = line.strip()
            if len(line) == 0:
                continue
            tokens = line.split(' ')
            embedding = np.empty([1, embedding_dim], dtype=np.float32)
            embedding[:] = tokens[1:]
            embedding_dict[tokens[0]] = embedding

    return embedding_dict, embedding_dim


def build_embedding_weight(vocab, embedding_dict, embedding_dim, caseless=True):
    scale = np.sqrt(3.0 / embedding_dim)
    embedding_weight = np.empty([len(vocab), embedding_dim], dtype=np.float32)
    for word, index in vocab.items():
        ww = word.lower() if caseless else word
        if ww in embedding_dict.keys():
            embedding = embedding_dict[ww]
        else:
            embedding = np.random.uniform(-scale, scale, [1, embedding_dim])
        embedding_weight[index, :] = embedding
    embedding_weight[0, :] = np.zeros([1, embedding_dim])  # pad 的初始向量为0，不让其进行参数传递
    return embedding_weight


def padding_sequences(examples, max_sent_num, max_sent_len, target_prompt=None):
    new_examples = []
    for idx, prompt, essay_ids, score, level in examples:
        if target_prompt:
            if int(prompt) == int(target_prompt):
                score = -1
        x = np.zeros(shape=(max_sent_num, max_sent_len), dtype=np.int32)
        sent_num = len(essay_ids)
        if sent_num > max_sent_num:
            sent_num = max_sent_num

        for i in range(sent_num):
            sent_ids = essay_ids[i]
            sent_len = len(sent_ids)
            for j in range(sent_len):
                word_id = sent_ids[j]
                x[i, j] = word_id

        new_examples.append([idx, prompt, x, score])
    return new_examples


def padding_sequences_multi_level(examples, max_sent_num, max_sent_len):
    new_examples = []
    for idx, prompt, essay_ids, score, level in examples:
        x = np.zeros(shape=(max_sent_num, max_sent_len), dtype=np.int32)
        sent_num = len(essay_ids)
        if sent_num > max_sent_num:
            sent_num = max_sent_num

        for i in range(sent_num):
            sent_ids = essay_ids[i]
            sent_len = len(sent_ids)
            for j in range(sent_len):
                word_id = sent_ids[j]
                x[i, j] = word_id

        new_examples.append([idx, prompt, x, score, level])
    return new_examples


def transfer_score(label, predict, prompt, mode):
    label = label.squeeze(-1).detach().numpy()
    predict = predict.squeeze(-1).detach().numpy()
    prompt = prompt.detach().numpy()
    transfer_label = []
    transfer_predict = []
    max_min_scores = score_range
    for i in range(len(prompt)):
        if mode == 'valid':
            s_r = [0, 10]
        else:
            s_r = max_min_scores[prompt[i]]
        transfer_label += [round(label[i] * (s_r[1]-s_r[0]) + s_r[0])]
        transfer_predict += [round(predict[i] * (s_r[1]-s_r[0]) + s_r[0])]

    return transfer_label, transfer_predict


def TransferScoreForSingleTrait(label, predict, prompt, mode, trait):
    label = label.squeeze(-1).detach().numpy()
    predict = predict.squeeze(-1).detach().numpy()
    prompt = prompt.detach().numpy()
    transfer_label = []
    transfer_predict = []
    max_min_scores = get_min_max_scores()
    for i in range(len(prompt)):
        if mode == 'valid':
            s_r = [0, 10]
        else:
            s_r = max_min_scores[prompt[i]][trait]
        transfer_label += [round(label[i] * (s_r[1]-s_r[0]) + s_r[0])]
        transfer_predict += [round(predict[i] * (s_r[1]-s_r[0]) + s_r[0])]

    return transfer_label, transfer_predict


def TransferScoreForSingleTrait_dev100(label, predict, prompt, mode, trait):
    label = label.squeeze(-1).detach().numpy()
    predict = predict.squeeze(-1).detach().numpy()
    prompt = prompt.detach().numpy()
    transfer_label = []
    transfer_predict = []
    max_min_scores = get_min_max_scores()
    for i in range(len(prompt)):
        if mode == 'valid':
            s_r = [0, 100]
        else:
            s_r = max_min_scores[prompt[i]][trait]
        transfer_label += [round(label[i] * (s_r[1]-s_r[0]) + s_r[0])]
        transfer_predict += [round(predict[i] * (s_r[1]-s_r[0]) + s_r[0])]

    return transfer_label, transfer_predict


def TransferScoreForMultiTrait(all_label, all_predict, label_list, predict_list, prompt_list, mode):
    label_list = label_list.detach().numpy()
    predict_list = predict_list.detach().numpy()
    prompt_list = prompt_list.detach().numpy()

    min_max_scores = get_min_max_scores()
    score_vector_positions = get_score_vector_positions()
    for index in range(len(prompt_list)):
        prompt = prompt_list[index]
        traits_name = min_max_scores[prompt].keys()
        label_line = label_list[index]
        predict_line = predict_list[index]
        for trait in traits_name:
            trait_position = score_vector_positions[trait]
            if label_line[trait_position] != -1:
                if mode == 'valid':
                    min_score, max_score = [0, 10]
                else:
                    min_score, max_score = min_max_scores[prompt][trait]

                trait_label = int(label_line[trait_position] * (max_score - min_score) + min_score)
                trait_predict = int(predict_line[trait_position] * (max_score - min_score) + min_score)
                all_label[trait].append(trait_label)
                all_predict[trait].append(trait_predict)

    return all_label, all_predict
