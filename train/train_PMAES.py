import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import random
import argparse
import numpy as np
import time
from tqdm import tqdm
from sklearn.metrics import f1_score

from metrics.metrics import kappa
from utils import utils
# from we_model_v1_reader import *

random.seed(111)
torch.manual_seed(111)
logger = utils.get_logger(name='Train...')


def mask_mse_loss_fn(label, predict):
    loss_fn = nn.MSELoss()
    mask_value = -1
    mask = torch.not_equal(label, mask_value)
    mse = loss_fn(label * mask, predict * mask)
    return mse


def mask_qwk(labels, predict, prompts, target_prompt):
    for index, prompt in enumerate(prompts):
        if prompt == target_prompt:
            labels[index] = 0
            predict[index] = 0
    return kappa(labels, predict, weights='quadratic') * 100


def GetAllEssayRepresentations(args, Gmodel, loader):
    prompt_essay_embed = torch.tensor([]).to('cpu')
    for item in loader:
        prompt, pos_ids, ling, read = item['prompt'], item['pos_ids'], item['ling'], item['read']
        with torch.no_grad():
            essay_embed = Gmodel(pos_ids.to(args.device)).to('cpu')
        prompt_essay_embed = torch.cat([prompt_essay_embed, essay_embed], dim=0)
    return prompt_essay_embed


def get_prompt_essay_embed_with_feature(args, Gmodel, loader):
    prompt_essay_embed = torch.tensor([]).to('cpu')
    for item in loader:
        prompt, pos_ids, ling, read = item['prompt'], item['pos_ids'], item['ling'], item['read']
        with torch.no_grad():
            essay_embed = Gmodel(pos_ids.to(args.device)).to('cpu')
            essay_embed = torch.cat([essay_embed, read, ling], dim=1)
        prompt_essay_embed = torch.cat([prompt_essay_embed, essay_embed], dim=0)
    return prompt_essay_embed


def TestSingleOverallScoring(args, essay_encoder, scorer, loader, mode, attribute_name):
    assert mode in ['valid', 'test']
    essay_encoder.eval()
    scorer.eval()

    aes_label_all = np.array([], dtype=int)
    aes_pre_all = np.array([], dtype=int)
    total_loss = 0.
    with torch.no_grad():
        for item in loader:
            prompt, essay, linguistic, readability, score = item['prompt'], item['pos_ids'], item['ling'], item['read'], item['score']
            essay_fea = essay_encoder(essay.to(args.device))
            fea_cat = torch.cat([essay_fea, linguistic.to(args.device), readability.to(args.device)], dim=1)
            aes_pre = scorer(fea_cat)
            aes_pre = aes_pre.to('cpu')
            aes_loss = nn.MSELoss()(aes_pre, score)

            total_loss += aes_loss
            score, aes_pre = utils.TransferScoreForSingleTrait(score, aes_pre, prompt, mode, attribute_name)
            aes_label_all = np.concatenate([aes_label_all, score])
            aes_pre_all = np.concatenate([aes_pre_all, aes_pre])
    qwk = kappa(aes_label_all, aes_pre_all, weights='quadratic') * 100
    loss = total_loss / len(loader)
    return qwk, loss


def TestSingleOverallScoringForMultiTarget(args, essay_encoder, Smodel, loader, mode, attribute_name):
    assert mode in ['valid', 'test']
    essay_encoder.eval()
    Smodel.eval()
    total_loss = 0.

    if mode == 'valid':
        aes_label_all = np.array([], dtype=int)
        aes_pre_all = np.array([], dtype=int)
        with torch.no_grad():
            for item in loader:
                prompt, essay, linguistic, readability, score = item['prompt'], item['pos_ids'], item['ling'], item['read'], item['score']
                essay_fea = essay_encoder(essay.to(args.device))
                fea_cat = torch.cat([essay_fea, linguistic.to(args.device), readability.to(args.device)], dim=1)
                aes_pre = Smodel(fea_cat)
                aes_pre = aes_pre.to('cpu')
                aes_loss = nn.MSELoss()(aes_pre, score)

                total_loss += aes_loss
                score, aes_pre = utils.TransferScoreForSingleTrait(score, aes_pre, prompt, mode, attribute_name)
                aes_label_all = np.concatenate([aes_label_all, score])
                aes_pre_all = np.concatenate([aes_pre_all, aes_pre])
        qwk = kappa(aes_label_all, aes_pre_all, weights='quadratic') * 100
        total_loss = total_loss / len(loader)
    else:
        aes_label_all = {}
        aes_pre_all = {}
        with torch.no_grad():
            for item in loader:
                prompt, essay, linguistic, readability, score = item['prompt'], item['pos_ids'], item['ling'], item['read'], item['score']
                essay_fea = essay_encoder(essay.to(args.device))
                fea_cat = torch.cat([essay_fea, linguistic.to(args.device), readability.to(args.device)], dim=1)
                aes_pre = Smodel(fea_cat)
                aes_pre = aes_pre.to('cpu')
                score, aes_pre = utils.TransferScoreForSingleTrait(score, aes_pre, prompt, mode, attribute_name)
                for index in range(len(prompt)):
                    p = prompt.numpy()[index]
                    aes_label_all[p] = aes_label_all.get(p, []) + [score[index]]
                    aes_pre_all[p] = aes_pre_all.get(p, []) + [aes_pre[index]]
        qwk = {}
        for key in aes_label_all.keys():
            score = aes_label_all[key]
            aes_pre = aes_pre_all[key]
            prompt_qwk = round(kappa(score, aes_pre, weights='quadratic') * 100, 2)
            qwk[key] = prompt_qwk
    return qwk, total_loss


def TestForSingleTrait_dev100(args, Gmodel, Smodel, loader, mode, attribute_name):
    assert mode in ['valid', 'test']
    Gmodel.eval()
    Smodel.eval()

    aes_label_all = np.array([], dtype=int)
    aes_pre_all = np.array([], dtype=int)
    total_loss = 0.
    with torch.no_grad():
        for item in loader:
            prompt, essay_ids, ling, read, aes_label = item['prompt'], item['pos_ids'], item['ling'], item['read'], item['score']
            essay_fea = Gmodel(essay_ids.to(args.device))
            fea_cat = torch.cat([essay_fea, ling.to(args.device), read.to(args.device)], dim=1)
            aes_pre = Smodel(fea_cat)
            aes_pre = aes_pre.to('cpu')
            aes_loss = nn.MSELoss()(aes_pre, aes_label)

            total_loss += aes_loss
            aes_label, aes_pre = utils.TransferScoreForSingleTrait_dev100(aes_label, aes_pre, prompt, mode, attribute_name)
            aes_label_all = np.concatenate([aes_label_all, aes_label])
            aes_pre_all = np.concatenate([aes_pre_all, aes_pre])
    qwk = kappa(aes_label_all, aes_pre_all, weights='quadratic') * 100
    loss = total_loss / len(loader)
    return qwk, loss


def TestForMultiTrait(args, Gmodel, Smodel, loader, mode, trait_interactive_type):
    assert mode in ['valid', 'test']
    Gmodel.eval()
    Smodel.eval()

    aes_label_all = {
        'score': [],
        'content': [],
        'organization': [],
        'word_choice': [],
        'sentence_fluency': [],
        'conventions': [],
        'prompt_adherence': [],
        'language': [],
        'narrativity': []
    }
    aes_pre_all = {
        'score': [],
        'content': [],
        'organization': [],
        'word_choice': [],
        'sentence_fluency': [],
        'conventions': [],
        'prompt_adherence': [],
        'language': [],
        'narrativity': []
    }
    with torch.no_grad():
        for item in loader:
            prompt, essay_ids, ling, read, aes_label = item['prompt'], item['pos_ids'], item['ling'], item['read'], item['score']
            essay_fea = Gmodel(essay_ids.to(args.device))
            # fea_cat = torch.cat([essay_fea, ling.to(args.device), read.to(args.device)], dim=1)
            if trait_interactive_type in ['attention', 'none']:
                aes_pre = Smodel(essay_fea, ling.to(args.device), read.to(args.device))
            else:
                aes_pre, _ = Smodel(essay_fea, ling.to(args.device), read.to(args.device))
            aes_pre = aes_pre.to('cpu')
            aes_label_all, aes_pre_all = utils.TransferScoreForMultiTrait(aes_label_all, aes_pre_all, aes_label, aes_pre, prompt, mode)

    qwk_result = {}
    average_qwk = 0
    for trait in aes_label_all.keys():
        label = aes_label_all[trait]
        predict = aes_pre_all[trait]
        if len(label) == 0:
            continue
        trait_qwk = kappa(label, predict, weights='quadratic') * 100
        qwk_result[trait] = trait_qwk
        average_qwk += trait_qwk

    average_qwk = np.mean(list(qwk_result.values()))
    qwk_result['Avg'] = average_qwk
    return qwk_result


def TrainSingleOverallScoring(args,
                              essay_encoder, scorer, pm_cl, optimizer,
                              tr_s_loader, va_s_loader, te_t_loader,
                              target_prompt_id, epoch,
                              tr_log, attribute_name):
    logger.info('Train other epoch: [TARGET] P:{} [EPOCH] E:{}...'.format(target_prompt_id, epoch))
    if epoch == 1:
        for item_index, s_item in tqdm(enumerate(tr_s_loader, start=1), desc='Training......'):
            essay_encoder.train(True)
            scorer.train(True)
            optimizer.zero_grad()
            s_prompt, s_pos_ids, s_ling, s_read, s_aes_label = s_item['prompt'], s_item['pos_ids'], s_item['ling'], \
                                                               s_item['read'], s_item['score']

            s_essay_fea = essay_encoder(s_pos_ids.to(args.device))
            s_fea_cat = torch.cat([s_essay_fea, s_ling.to(args.device), s_read.to(args.device)], dim=1)
            s_aes_pre = scorer(s_fea_cat)
            s_aes_pre = s_aes_pre.to('cpu')
            aes_loss = nn.MSELoss()(s_aes_pre, s_aes_label)
            aes_loss.backward(retain_graph=True)
            optimizer.step()
    else:
        s_essay_embed = GetAllEssayRepresentations(args, essay_encoder, tr_s_loader)
        t_essay_embed = GetAllEssayRepresentations(args, essay_encoder, te_t_loader)
        for item_index, (s_item, t_item) in tqdm(enumerate(zip(tr_s_loader, te_t_loader), start=1), desc='Training......'):
            essay_encoder.train(True)
            scorer.train(True)
            pm_cl.train(True)
            optimizer.zero_grad()

            s_prompt, s_pos_ids, s_ling, s_read, s_aes_label = s_item['prompt'], s_item['pos_ids'], s_item['ling'], s_item['read'], s_item['score']
            t_prompt, t_pos_ids, t_ling, t_read, t_aes_label = t_item['prompt'], t_item['pos_ids'], t_item['ling'], t_item['read'], t_item['score']

            # Start First Step
            s_essay_fea_1 = essay_encoder(s_pos_ids.to(args.device))
            t_essay_fea_1 = essay_encoder(t_pos_ids.to(args.device))
            cl_loss_1 = 0.5 * pm_cl(s_essay_fea_1, t_essay_fea_1, s_essay_embed.to(args.device), t_essay_embed.to(args.device))
            first_loss = cl_loss_1
            first_loss.backward(retain_graph=True)
            optimizer.step()
            # End First Step

            s_essay_fea_2 = essay_encoder(s_pos_ids.to(args.device))
            t_essay_fea_2 = essay_encoder(t_pos_ids.to(args.device))
            s_fea_cat_2 = torch.cat([s_essay_fea_2, s_ling.to(args.device), s_read.to(args.device)], dim=1)
            s_aes_pre_2 = scorer(s_fea_cat_2)
            s_aes_pre_2 = s_aes_pre_2.to('cpu')
            aes_loss_2 = nn.MSELoss()(s_aes_pre_2, s_aes_label)

            cl_loss_2 = 0.5 * pm_cl(s_essay_fea_2, t_essay_fea_2, s_essay_embed.to(args.device), t_essay_embed.to(args.device))
            second_loss = aes_loss_2 + cl_loss_2
            second_loss.backward(retain_graph=True)
            optimizer.step()

            s_essay_fea_3 = essay_encoder(s_pos_ids.to(args.device))
            s_fea_cat_3 = torch.cat([s_essay_fea_3, s_ling.to(args.device), s_read.to(args.device)], dim=1)
            s_aes_pre_3 = scorer(s_fea_cat_3)
            s_aes_pre_3 = s_aes_pre_3.to('cpu')
            aes_loss_3 = nn.MSELoss()(s_aes_pre_3, s_aes_label)
            third_loss = aes_loss_3
            third_loss.backward(retain_graph=True)
            optimizer.step()
    if args.source2target == 'many2one' or args.source2target == 'one2one':
        va_qwk, va_loss = TestSingleOverallScoring(args, essay_encoder, scorer, va_s_loader, 'valid', attribute_name)
        te_qwk, te_loss = TestSingleOverallScoring(args, essay_encoder, scorer, te_t_loader, 'test', attribute_name)
        if va_qwk > tr_log['Epoch_best_dev_qwk'][0]:
            tr_log['Epoch_best_dev_qwk'][0] = va_qwk
            tr_log['Epoch_best_dev_qwk'][1] = te_qwk
            tr_log['Epoch_best_dev_qwk'][2] = epoch
        epoch_msg = '[EPOCH DEV] QWK:{:.2f}, [EPOCH TEST] QWK:{:.2f}, [BEST DEV IN TEST] QWK:{:.2f}'
        epoch_msg = epoch_msg.format(va_qwk, te_qwk, tr_log['Epoch_best_dev_qwk'][1])
        logger.info(epoch_msg)

    else:  #  args.source2target == 'many2many' or args.source2target == 'one2many'
        va_qwk, va_loss = TestSingleOverallScoringForMultiTarget(args, essay_encoder, scorer, va_s_loader, 'valid', attribute_name)
        te_qwk, te_loss = TestSingleOverallScoringForMultiTarget(args, essay_encoder, scorer, te_t_loader, 'test', attribute_name)
        if va_qwk > tr_log['Epoch_best_dev_qwk'][0]:
            tr_log['Epoch_best_dev_qwk'][0] = va_qwk
            tr_log['Epoch_best_dev_qwk'][1] = str(te_qwk)
            tr_log['Epoch_best_dev_qwk'][2] = epoch

        epoch_msg = '[EPOCH DEV] QWK:{:.2f}, [EPOCH TEST] QWK:{}, [BEST DEV IN TEST] QWK:{}'
        epoch_msg = epoch_msg.format(va_qwk, te_qwk, tr_log['Epoch_best_dev_qwk'][1])
        logger.info(epoch_msg)



def TrainSingleOverallScoringForMultiTarget(args,
                                            Gmodel, Smodel, FCmodel, optims,
                                            tr_s_loader, va_s_loader, te_t_loader,
                                            t_index, e_index,
                                            tr_log, attribute_name):

    logger.info('Train other epoch: [TARGET] P:{} [EPOCH] E:{}...'.format(t_index, e_index))
    if e_index == 1:
        for item_index, s_item in tqdm(enumerate(tr_s_loader, start=1), desc='Training......'):
            Gmodel.train(True)
            Smodel.train(True)
            optims.zero_grad()
            s_prompt, s_pos_ids, s_ling, s_read, s_aes_label = s_item['prompt'], s_item['pos_ids'], s_item['ling'], \
                                                               s_item['read'], s_item['score']

            s_essay_fea = Gmodel(s_pos_ids.to(args.device))
            s_fea_cat = torch.cat([s_essay_fea, s_ling.to(args.device), s_read.to(args.device)], dim=1)
            s_aes_pre = Smodel(s_fea_cat)
            s_aes_pre = s_aes_pre.to('cpu')
            aes_loss = nn.MSELoss()(s_aes_pre, s_aes_label)
            aes_loss.backward(retain_graph=True)
            optims.step()
    else:
        s_essay_embed = GetAllEssayRepresentations(args, Gmodel, tr_s_loader)
        t_essay_embed = GetAllEssayRepresentations(args, Gmodel, te_t_loader)
        for item_index, (s_item, t_item) in tqdm(enumerate(zip(tr_s_loader, te_t_loader), start=1), desc='Training......'):
            Gmodel.train(True)
            Smodel.train(True)
            FCmodel.train(True)
            optims.zero_grad()

            s_prompt, s_pos_ids, s_ling, s_read, s_aes_label = s_item['prompt'], s_item['pos_ids'], s_item['ling'], s_item['read'], s_item['score']
            t_prompt, t_pos_ids, t_ling, t_read, t_aes_label = t_item['prompt'], t_item['pos_ids'], t_item['ling'], t_item['read'], t_item['score']

            # Start First Step
            s_essay_fea_1 = Gmodel(s_pos_ids.to(args.device))
            t_essay_fea_1 = Gmodel(t_pos_ids.to(args.device))
            cl_loss_1 = 0.5 * FCmodel(s_essay_fea_1, t_essay_fea_1, s_essay_embed.to(args.device), t_essay_embed.to(args.device))
            first_loss = cl_loss_1
            first_loss.backward(retain_graph=True)
            optims.step()
            # End First Step

            s_essay_fea_2 = Gmodel(s_pos_ids.to(args.device))
            t_essay_fea_2 = Gmodel(t_pos_ids.to(args.device))
            s_fea_cat_2 = torch.cat([s_essay_fea_2, s_ling.to(args.device), s_read.to(args.device)], dim=1)
            s_aes_pre_2 = Smodel(s_fea_cat_2)
            s_aes_pre_2 = s_aes_pre_2.to('cpu')
            aes_loss_2 = nn.MSELoss()(s_aes_pre_2, s_aes_label)

            cl_loss_2 = 0.5 * FCmodel(s_essay_fea_2, t_essay_fea_2, s_essay_embed.to(args.device), t_essay_embed.to(args.device))
            second_loss = aes_loss_2 + cl_loss_2
            second_loss.backward(retain_graph=True)
            optims.step()

            s_essay_fea_3 = Gmodel(s_pos_ids.to(args.device))
            s_fea_cat_3 = torch.cat([s_essay_fea_3, s_ling.to(args.device), s_read.to(args.device)], dim=1)
            s_aes_pre_3 = Smodel(s_fea_cat_3)
            s_aes_pre_3 = s_aes_pre_3.to('cpu')
            aes_loss_3 = nn.MSELoss()(s_aes_pre_3, s_aes_label)
            third_loss = aes_loss_3
            third_loss.backward(retain_graph=True)
            optims.step()

    va_qwk, va_loss = TestSingleOverallScoringForMultiTarget(args, Gmodel, Smodel, va_s_loader, 'valid', attribute_name)
    te_qwk, te_loss = TestSingleOverallScoringForMultiTarget(args, Gmodel, Smodel, te_t_loader, 'test', attribute_name)
    if va_qwk > tr_log['Epoch_best_dev_qwk'][0]:
        tr_log['Epoch_best_dev_qwk'][0] = va_qwk
        tr_log['Epoch_best_dev_qwk'][1] = str(te_qwk)
        tr_log['Epoch_best_dev_qwk'][2] = e_index

    epoch_msg = '[EPOCH DEV] QWK:{:.2f}, [EPOCH TEST] QWK:{}, [BEST DEV IN TEST] QWK:{}'
    epoch_msg = epoch_msg.format(va_qwk, te_qwk, tr_log['Epoch_best_dev_qwk'][1])
    logger.info(epoch_msg)


def TrainForSingleTraitDoublePCLDirectly(args,
                                 Gmodel, Smodel, FCmodel, optims,
                                 tr_s_loader, va_s_loader, te_t_loader,
                                 t_index, e_index,
                                 tr_log, attribute_name):
    logger.info('Train other epoch: [TARGET] P:{} [EPOCH] E:{}...'.format(t_index, e_index))
    s_essay_embed = GetAllEssayRepresentations(args, Gmodel, tr_s_loader)
    t_essay_embed = GetAllEssayRepresentations(args, Gmodel, te_t_loader)
    for item_index, (s_item, t_item) in tqdm(enumerate(zip(tr_s_loader, te_t_loader), start=1), desc='Training......'):
        Gmodel.train(True)
        Smodel.train(True)
        FCmodel.train(True)
        optims.zero_grad()

        s_prompt, s_pos_ids, s_ling, s_read, s_aes_label = s_item['prompt'], s_item['pos_ids'], s_item['ling'], s_item['read'], s_item['score']
        t_prompt, t_pos_ids, t_ling, t_read, t_aes_label = t_item['prompt'], t_item['pos_ids'], t_item['ling'], t_item['read'], t_item['score']

        # Start First Step
        s_essay_fea_1 = Gmodel(s_pos_ids.to(args.device))
        t_essay_fea_1 = Gmodel(t_pos_ids.to(args.device))
        cl_loss_1 = 0.5 * FCmodel(s_essay_fea_1, t_essay_fea_1, s_essay_embed.to(args.device), t_essay_embed.to(args.device))
        first_loss = cl_loss_1
        first_loss.backward(retain_graph=True)
        optims.step()
        # End First Step

        s_essay_fea_2 = Gmodel(s_pos_ids.to(args.device))
        t_essay_fea_2 = Gmodel(t_pos_ids.to(args.device))
        s_fea_cat_2 = torch.cat([s_essay_fea_2, s_ling.to(args.device), s_read.to(args.device)], dim=1)
        s_aes_pre_2 = Smodel(s_fea_cat_2)
        s_aes_pre_2 = s_aes_pre_2.to('cpu')
        aes_loss_2 = nn.MSELoss()(s_aes_pre_2, s_aes_label)

        cl_loss_2 = 0.5 * FCmodel(s_essay_fea_2, t_essay_fea_2, s_essay_embed.to(args.device), t_essay_embed.to(args.device))
        second_loss = aes_loss_2 + cl_loss_2
        second_loss.backward(retain_graph=True)
        optims.step()

        s_essay_fea_3 = Gmodel(s_pos_ids.to(args.device))
        s_fea_cat_3 = torch.cat([s_essay_fea_3, s_ling.to(args.device), s_read.to(args.device)], dim=1)
        s_aes_pre_3 = Smodel(s_fea_cat_3)
        s_aes_pre_3 = s_aes_pre_3.to('cpu')
        aes_loss_3 = nn.MSELoss()(s_aes_pre_3, s_aes_label)
        third_loss = aes_loss_3
        third_loss.backward(retain_graph=True)
        optims.step()

    va_qwk, va_loss = TestSingleOverallScoring(args, Gmodel, Smodel, va_s_loader, 'valid', attribute_name)
    te_qwk, te_loss = TestSingleOverallScoring(args, Gmodel, Smodel, te_t_loader, 'test', attribute_name)
    if va_qwk > tr_log['Epoch_best_dev_qwk'][0]:
        tr_log['Epoch_best_dev_qwk'][0] = va_qwk
        tr_log['Epoch_best_dev_qwk'][1] = te_qwk
        tr_log['Epoch_best_dev_qwk'][2] = e_index
        tr_log['BestModel']['EpochBestGmodel'] = Gmodel.state_dict()
        tr_log['BestModel']['EpochBestSmodel'] = Smodel.state_dict()

    if va_loss < tr_log['Epoch_lowest_dev_loss'][0]:
        tr_log['Epoch_lowest_dev_loss'][0] = va_loss
        tr_log['Epoch_lowest_dev_loss'][1] = te_qwk
        tr_log['Epoch_lowest_dev_loss'][2] = e_index
    epoch_msg = '[EPOCH DEV] QWK:{:.2f}, [EPOCH TEST] QWK:{:.2f}, [BEST DEV IN TEST] QWK:{:.2f}, [LOWEST DEV IN TEST] QWK:{:.2f}'
    epoch_msg = epoch_msg.format(va_qwk, te_qwk, tr_log['Epoch_best_dev_qwk'][1], tr_log['Epoch_lowest_dev_loss'][1])
    logger.info(epoch_msg)

    epoch_final_msg = '[TARGET] P:{} [EPOCH] E:{} [BATCH BEST DEV IN TEST] QWK:{:.2f} [BATCH LOWEST DEV IN TEST] QWK:{:.2f}'
    epoch_final_msg = epoch_final_msg.format(t_index, e_index, tr_log['Best_dev_qwk'][1], tr_log['Lowest_dev_loss'][1])
    logger.info(epoch_final_msg)


def TrainForSingleTraitDoublePCL_dev100(args,
                                 Gmodel, Smodel, FCmodel, optims,
                                 tr_s_loader, va_s_loader, te_t_loader,
                                 t_index, e_index,
                                 tr_log, attribute_name):
    logger.info('Train other epoch: [TARGET] P:{} [EPOCH] E:{}...'.format(t_index, e_index))
    if e_index == 1:
        for item_index, s_item in tqdm(enumerate(tr_s_loader, start=1), desc='Training......'):
            Gmodel.train(True)
            Smodel.train(True)
            optims.zero_grad()
            s_prompt, s_pos_ids, s_ling, s_read, s_aes_label = s_item['prompt'], s_item['pos_ids'], s_item['ling'], \
                                                               s_item['read'], s_item['score']

            s_essay_fea = Gmodel(s_pos_ids.to(args.device))
            s_fea_cat = torch.cat([s_essay_fea, s_ling.to(args.device), s_read.to(args.device)], dim=1)
            s_aes_pre = Smodel(s_fea_cat)
            s_aes_pre = s_aes_pre.to('cpu')
            aes_loss = nn.MSELoss()(s_aes_pre, s_aes_label)
            aes_loss.backward(retain_graph=True)
            optims.step()
    else:
        s_essay_embed = GetAllEssayRepresentations(args, Gmodel, tr_s_loader)
        t_essay_embed = GetAllEssayRepresentations(args, Gmodel, te_t_loader)
        for item_index, (s_item, t_item) in tqdm(enumerate(zip(tr_s_loader, te_t_loader), start=1), desc='Training......'):
            Gmodel.train(True)
            Smodel.train(True)
            FCmodel.train(True)
            optims.zero_grad()

            s_prompt, s_pos_ids, s_ling, s_read, s_aes_label = s_item['prompt'], s_item['pos_ids'], s_item['ling'], s_item['read'], s_item['score']
            t_prompt, t_pos_ids, t_ling, t_read, t_aes_label = t_item['prompt'], t_item['pos_ids'], t_item['ling'], t_item['read'], t_item['score']

            # Start First Step
            s_essay_fea_1 = Gmodel(s_pos_ids.to(args.device))
            t_essay_fea_1 = Gmodel(t_pos_ids.to(args.device))
            cl_loss_1 = 0.5 * FCmodel(s_essay_fea_1, t_essay_fea_1, s_essay_embed.to(args.device), t_essay_embed.to(args.device))
            first_loss = cl_loss_1
            first_loss.backward(retain_graph=True)
            optims.step()
            # End First Step

            s_essay_fea_2 = Gmodel(s_pos_ids.to(args.device))
            t_essay_fea_2 = Gmodel(t_pos_ids.to(args.device))
            s_fea_cat_2 = torch.cat([s_essay_fea_2, s_ling.to(args.device), s_read.to(args.device)], dim=1)
            s_aes_pre_2 = Smodel(s_fea_cat_2)
            s_aes_pre_2 = s_aes_pre_2.to('cpu')
            aes_loss_2 = nn.MSELoss()(s_aes_pre_2, s_aes_label)

            cl_loss_2 = 0.5 * FCmodel(s_essay_fea_2, t_essay_fea_2, s_essay_embed.to(args.device), t_essay_embed.to(args.device))
            second_loss = aes_loss_2 + cl_loss_2
            second_loss.backward(retain_graph=True)
            optims.step()

            s_essay_fea_3 = Gmodel(s_pos_ids.to(args.device))
            s_fea_cat_3 = torch.cat([s_essay_fea_3, s_ling.to(args.device), s_read.to(args.device)], dim=1)
            s_aes_pre_3 = Smodel(s_fea_cat_3)
            s_aes_pre_3 = s_aes_pre_3.to('cpu')
            aes_loss_3 = nn.MSELoss()(s_aes_pre_3, s_aes_label)
            third_loss = aes_loss_3
            third_loss.backward(retain_graph=True)
            optims.step()

    va_qwk, va_loss = TestForSingleTrait_dev100(args, Gmodel, Smodel, va_s_loader, 'valid', attribute_name)
    te_qwk, te_loss = TestForSingleTrait_dev100(args, Gmodel, Smodel, te_t_loader, 'test', attribute_name)
    if va_qwk > tr_log['Epoch_best_dev_qwk'][0]:
        tr_log['Epoch_best_dev_qwk'][0] = va_qwk
        tr_log['Epoch_best_dev_qwk'][1] = te_qwk
        tr_log['Epoch_best_dev_qwk'][2] = e_index
        tr_log['BestModel']['EpochBestGmodel'] = Gmodel.state_dict()
        tr_log['BestModel']['EpochBestSmodel'] = Smodel.state_dict()

    if va_loss < tr_log['Epoch_lowest_dev_loss'][0]:
        tr_log['Epoch_lowest_dev_loss'][0] = va_loss
        tr_log['Epoch_lowest_dev_loss'][1] = te_qwk
        tr_log['Epoch_lowest_dev_loss'][2] = e_index
    epoch_msg = '[EPOCH DEV] QWK:{:.2f}, [EPOCH TEST] QWK:{:.2f}, [BEST DEV IN TEST] QWK:{:.2f}, [LOWEST DEV IN TEST] QWK:{:.2f}'
    epoch_msg = epoch_msg.format(va_qwk, te_qwk, tr_log['Epoch_best_dev_qwk'][1], tr_log['Epoch_lowest_dev_loss'][1])
    logger.info(epoch_msg)

    epoch_final_msg = '[TARGET] P:{} [EPOCH] E:{} [BATCH BEST DEV IN TEST] QWK:{:.2f} [BATCH LOWEST DEV IN TEST] QWK:{:.2f}'
    epoch_final_msg = epoch_final_msg.format(t_index, e_index, tr_log['Best_dev_qwk'][1], tr_log['Lowest_dev_loss'][1])
    logger.info(epoch_final_msg)


def TrainForSingleTraitSourcePCL(args,
                                 Gmodel, Smodel, FCmodel, optims,
                                 tr_s_loader, va_s_loader, te_t_loader,
                                 t_index, e_index,
                                 tr_log, attribute_name):
    logger.info('Train other epoch: [TARGET] P:{} [EPOCH] E:{}...'.format(t_index, e_index))
    if e_index == 1:
        for item_index, s_item in tqdm(enumerate(tr_s_loader, start=1), desc='Training......'):
            Gmodel.train(True)
            Smodel.train(True)
            optims.zero_grad()

            s_prompt, s_pos_ids, s_ling, s_read, s_aes_label = s_item['prompt'], s_item['pos_ids'], s_item['ling'], \
                                                               s_item['read'], s_item['score']

            s_essay_fea = Gmodel(s_pos_ids.to(args.device))
            s_fea_cat = torch.cat([s_essay_fea, s_ling.to(args.device), s_read.to(args.device)], dim=1)
            s_aes_pre = Smodel(s_fea_cat)
            s_aes_pre = s_aes_pre.to('cpu')
            aes_loss = nn.MSELoss()(s_aes_pre, s_aes_label)
            aes_loss.backward(retain_graph=True)
            optims.step()
    else:
        t_essay_embed = GetAllEssayRepresentations(args, Gmodel, te_t_loader)
        for item_index, s_item in tqdm(enumerate(tr_s_loader, start=1), desc='Training......'):
            Gmodel.train(True)
            Smodel.train(True)
            FCmodel.train(True)
            optims.zero_grad()

            s_prompt, s_pos_ids, s_ling, s_read, s_aes_label = s_item['prompt'], s_item['pos_ids'], s_item['ling'], s_item['read'], s_item['score']

            # Start First Step
            s_essay_fea_1 = Gmodel(s_pos_ids.to(args.device))
            cl_loss_1 = 0.5 * FCmodel(s_essay_fea_1, t_essay_embed.to(args.device))
            first_loss = cl_loss_1
            first_loss.backward(retain_graph=True)
            optims.step()
            # End First Step

            s_essay_fea_2 = Gmodel(s_pos_ids.to(args.device))
            s_fea_cat_2 = torch.cat([s_essay_fea_2, s_ling.to(args.device), s_read.to(args.device)], dim=1)
            s_aes_pre_2 = Smodel(s_fea_cat_2)
            s_aes_pre_2 = s_aes_pre_2.to('cpu')
            aes_loss_2 = nn.MSELoss()(s_aes_pre_2, s_aes_label)

            cl_loss_2 = 0.5 * FCmodel(s_essay_fea_2, t_essay_embed.to(args.device))
            second_loss = aes_loss_2 + cl_loss_2
            second_loss.backward(retain_graph=True)
            optims.step()

            s_essay_fea_3 = Gmodel(s_pos_ids.to(args.device))
            s_fea_cat_3 = torch.cat([s_essay_fea_3, s_ling.to(args.device), s_read.to(args.device)], dim=1)
            s_aes_pre_3 = Smodel(s_fea_cat_3)
            s_aes_pre_3 = s_aes_pre_3.to('cpu')
            aes_loss_3 = nn.MSELoss()(s_aes_pre_3, s_aes_label)
            third_loss = aes_loss_3
            third_loss.backward(retain_graph=True)
            optims.step()

    va_qwk, va_loss = TestSingleOverallScoring(args, Gmodel, Smodel, va_s_loader, 'valid', attribute_name)
    te_qwk, te_loss = TestSingleOverallScoring(args, Gmodel, Smodel, te_t_loader, 'test', attribute_name)
    if va_qwk > tr_log['Epoch_best_dev_qwk'][0]:
        tr_log['Epoch_best_dev_qwk'][0] = va_qwk
        tr_log['Epoch_best_dev_qwk'][1] = te_qwk
        tr_log['Epoch_best_dev_qwk'][2] = e_index
        tr_log['BestModel']['EpochBestGmodel'] = Gmodel.state_dict()
        tr_log['BestModel']['EpochBestSmodel'] = Smodel.state_dict()

    if va_loss < tr_log['Epoch_lowest_dev_loss'][0]:
        tr_log['Epoch_lowest_dev_loss'][0] = va_loss
        tr_log['Epoch_lowest_dev_loss'][1] = te_qwk
        tr_log['Epoch_lowest_dev_loss'][2] = e_index
    epoch_msg = '[EPOCH DEV] QWK:{:.2f}, [EPOCH TEST] QWK:{:.2f}, [BEST DEV IN TEST] QWK:{:.2f}, [LOWEST DEV IN TEST] QWK:{:.2f}'
    epoch_msg = epoch_msg.format(va_qwk, te_qwk, tr_log['Epoch_best_dev_qwk'][1], tr_log['Epoch_lowest_dev_loss'][1])
    logger.info(epoch_msg)
    epoch_final_msg = '[TARGET] P:{} [EPOCH] E:{} [BATCH BEST DEV IN TEST] QWK:{:.2f} [BATCH LOWEST DEV IN TEST] QWK:{:.2f}'
    epoch_final_msg = epoch_final_msg.format(t_index, e_index, tr_log['Best_dev_qwk'][1], tr_log['Lowest_dev_loss'][1])
    logger.info(epoch_final_msg)

def TrainForSingleTraitTargetPCL(args,
                                 Gmodel, Smodel, FCmodel, optims,
                                 tr_s_loader, va_s_loader, te_t_loader,
                                 t_index, e_index,
                                 tr_log, attribute_name):
    logger.info('Train other epoch: [TARGET] P:{} [EPOCH] E:{}...'.format(t_index, e_index))
    if e_index == 1:
        for item_index, s_item in tqdm(enumerate(tr_s_loader, start=1), desc='Training......'):
            Gmodel.train(True)
            Smodel.train(True)
            optims.zero_grad()

            s_prompt, s_pos_ids, s_ling, s_read, s_aes_label = s_item['prompt'], s_item['pos_ids'], s_item['ling'], \
                                                               s_item['read'], s_item['score']

            s_essay_fea = Gmodel(s_pos_ids.to(args.device))
            s_fea_cat = torch.cat([s_essay_fea, s_ling.to(args.device), s_read.to(args.device)], dim=1)
            s_aes_pre = Smodel(s_fea_cat)
            s_aes_pre = s_aes_pre.to('cpu')
            aes_loss = nn.MSELoss()(s_aes_pre, s_aes_label)
            aes_loss.backward(retain_graph=True)
            optims.step()
    else:
        s_essay_embed = GetAllEssayRepresentations(args, Gmodel, tr_s_loader)
        for item_index, (s_item, t_item) in tqdm(enumerate(zip(tr_s_loader, te_t_loader), start=1), desc='Training......'):
            Gmodel.train(True)
            Smodel.train(True)
            FCmodel.train(True)
            optims.zero_grad()

            s_prompt, s_pos_ids, s_ling, s_read, s_aes_label = s_item['prompt'], s_item['pos_ids'], s_item['ling'], s_item['read'], s_item['score']
            t_prompt, t_pos_ids, t_ling, t_read, t_aes_label = t_item['prompt'], t_item['pos_ids'], t_item['ling'], t_item['read'], t_item['score']

            # Start First Step
            t_essay_fea_1 = Gmodel(t_pos_ids.to(args.device))
            cl_loss_1 = 0.5 * FCmodel(t_essay_fea_1, s_essay_embed.to(args.device))
            first_loss = cl_loss_1
            first_loss.backward(retain_graph=True)
            optims.step()
            # End First Step

            s_essay_fea_2 = Gmodel(s_pos_ids.to(args.device))

            s_fea_cat_2 = torch.cat([s_essay_fea_2, s_ling.to(args.device), s_read.to(args.device)], dim=1)
            s_aes_pre_2 = Smodel(s_fea_cat_2)
            s_aes_pre_2 = s_aes_pre_2.to('cpu')
            aes_loss_2 = nn.MSELoss()(s_aes_pre_2, s_aes_label)

            t_essay_fea_2 = Gmodel(t_pos_ids.to(args.device))
            cl_loss_2 = 0.5 * FCmodel(t_essay_fea_2, s_essay_embed.to(args.device))
            second_loss = aes_loss_2 + cl_loss_2
            second_loss.backward(retain_graph=True)
            optims.step()

            s_essay_fea_3 = Gmodel(s_pos_ids.to(args.device))
            s_fea_cat_3 = torch.cat([s_essay_fea_3, s_ling.to(args.device), s_read.to(args.device)], dim=1)
            s_aes_pre_3 = Smodel(s_fea_cat_3)
            s_aes_pre_3 = s_aes_pre_3.to('cpu')
            aes_loss_3 = nn.MSELoss()(s_aes_pre_3, s_aes_label)
            third_loss = aes_loss_3
            third_loss.backward(retain_graph=True)
            optims.step()

    va_qwk, va_loss = TestSingleOverallScoring(args, Gmodel, Smodel, va_s_loader, 'valid', attribute_name)
    te_qwk, te_loss = TestSingleOverallScoring(args, Gmodel, Smodel, te_t_loader, 'test', attribute_name)
    if va_qwk > tr_log['Epoch_best_dev_qwk'][0]:
        tr_log['Epoch_best_dev_qwk'][0] = va_qwk
        tr_log['Epoch_best_dev_qwk'][1] = te_qwk
        tr_log['Epoch_best_dev_qwk'][2] = e_index
        tr_log['BestModel']['EpochBestGmodel'] = Gmodel.state_dict()
        tr_log['BestModel']['EpochBestSmodel'] = Smodel.state_dict()

    if va_loss < tr_log['Epoch_lowest_dev_loss'][0]:
        tr_log['Epoch_lowest_dev_loss'][0] = va_loss
        tr_log['Epoch_lowest_dev_loss'][1] = te_qwk
        tr_log['Epoch_lowest_dev_loss'][2] = e_index
    epoch_msg = '[EPOCH DEV] QWK:{:.2f}, [EPOCH TEST] QWK:{:.2f}, [BEST DEV IN TEST] QWK:{:.2f}, [LOWEST DEV IN TEST] QWK:{:.2f}'
    epoch_msg = epoch_msg.format(va_qwk, te_qwk, tr_log['Epoch_best_dev_qwk'][1], tr_log['Epoch_lowest_dev_loss'][1])
    logger.info(epoch_msg)


def TrainForSingleTraitNoPCL(args,
                             Gmodel, Smodel, optims,
                             tr_s_loader, va_s_loader, te_t_loader,
                             t_index, e_index,
                             tr_log, attribute_name):
    logger.info('Train other epoch: [TARGET] P:{} [EPOCH] E:{}...'.format(t_index, e_index))
    for item_index, s_item in tqdm(enumerate(tr_s_loader, start=1), desc='Training......'):
        Gmodel.train(True)
        Smodel.train(True)
        optims.zero_grad()

        s_prompt, s_pos_ids, s_ling, s_read, s_aes_label = s_item['prompt'], s_item['pos_ids'], s_item['ling'], \
                                                           s_item['read'], s_item['score']

        s_essay_fea = Gmodel(s_pos_ids.to(args.device))
        s_fea_cat = torch.cat([s_essay_fea, s_ling.to(args.device), s_read.to(args.device)], dim=1)
        s_aes_pre = Smodel(s_fea_cat)
        s_aes_pre = s_aes_pre.to('cpu')
        aes_loss = nn.MSELoss()(s_aes_pre, s_aes_label)
        aes_loss.backward(retain_graph=True)
        optims.step()

    va_qwk, va_loss = TestSingleOverallScoring(args, Gmodel, Smodel, va_s_loader, 'valid', attribute_name)
    te_qwk, te_loss = TestSingleOverallScoring(args, Gmodel, Smodel, te_t_loader, 'test', attribute_name)
    if va_qwk > tr_log['Epoch_best_dev_qwk'][0]:
        tr_log['Epoch_best_dev_qwk'][0] = va_qwk
        tr_log['Epoch_best_dev_qwk'][1] = te_qwk
        tr_log['Epoch_best_dev_qwk'][2] = e_index
        tr_log['BestModel']['EpochBestGmodel'] = Gmodel.state_dict()
        tr_log['BestModel']['EpochBestSmodel'] = Smodel.state_dict()

    if va_loss < tr_log['Epoch_lowest_dev_loss'][0]:
        tr_log['Epoch_lowest_dev_loss'][0] = va_loss
        tr_log['Epoch_lowest_dev_loss'][1] = te_qwk
        tr_log['Epoch_lowest_dev_loss'][2] = e_index
    epoch_msg = '[EPOCH DEV] QWK:{:.2f}, [EPOCH TEST] QWK:{:.2f}, [BEST DEV IN TEST] QWK:{:.2f}, [LOWEST DEV IN TEST] QWK:{:.2f}'
    epoch_msg = epoch_msg.format(va_qwk, te_qwk, tr_log['Epoch_best_dev_qwk'][1], tr_log['Epoch_lowest_dev_loss'][1])
    logger.info(epoch_msg)
    epoch_final_msg = '[TARGET] P:{} [EPOCH] E:{} [BATCH BEST DEV IN TEST] QWK:{:.2f} [BATCH LOWEST DEV IN TEST] QWK:{:.2f}'
    epoch_final_msg = epoch_final_msg.format(t_index, e_index, tr_log['Best_dev_qwk'][1], tr_log['Lowest_dev_loss'][1])
    logger.info(epoch_final_msg)


def TrainForSingleTraitNoPCLMultiTest(args,
                                      Gmodel, Smodel, optims,
                                      tr_s_loader, va_s_loader, te_t_loader,
                                      t_index, e_index,
                                      tr_log, attribute_name):
    logger.info('Train other epoch: [TARGET] P:{} [EPOCH] E:{}...'.format(t_index, e_index))
    for item_index, s_item in tqdm(enumerate(tr_s_loader, start=1), desc='Training......'):
        Gmodel.train(True)
        Smodel.train(True)
        optims.zero_grad()

        s_prompt, s_pos_ids, s_ling, s_read, s_aes_label = s_item['prompt'], s_item['pos_ids'], s_item['ling'], \
                                                           s_item['read'], s_item['score']

        s_essay_fea = Gmodel(s_pos_ids.to(args.device))
        s_fea_cat = torch.cat([s_essay_fea, s_ling.to(args.device), s_read.to(args.device)], dim=1)
        s_aes_pre = Smodel(s_fea_cat)
        s_aes_pre = s_aes_pre.to('cpu')
        aes_loss = nn.MSELoss()(s_aes_pre, s_aes_label)
        aes_loss.backward(retain_graph=True)
        optims.step()

    va_qwk, va_loss = TestSingleOverallScoringForMultiTarget(args, Gmodel, Smodel, va_s_loader, 'valid', attribute_name)
    te_qwk, te_loss = TestSingleOverallScoringForMultiTarget(args, Gmodel, Smodel, te_t_loader, 'test', attribute_name)
    if va_qwk > tr_log['Epoch_best_dev_qwk'][0]:
        tr_log['Epoch_best_dev_qwk'][0] = va_qwk
        tr_log['Epoch_best_dev_qwk'][1] = str(te_qwk)
        tr_log['Epoch_best_dev_qwk'][2] = e_index
        tr_log['BestModel']['EpochBestGmodel'] = Gmodel.state_dict()
        tr_log['BestModel']['EpochBestSmodel'] = Smodel.state_dict()

    if va_loss < tr_log['Epoch_lowest_dev_loss'][0]:
        tr_log['Epoch_lowest_dev_loss'][0] = va_loss
        tr_log['Epoch_lowest_dev_loss'][1] = str(te_qwk)
        tr_log['Epoch_lowest_dev_loss'][2] = e_index
    epoch_msg = '[EPOCH DEV] QWK:{:.2f}, [EPOCH TEST] QWK:{}, [BEST DEV IN TEST] QWK:{}, [LOWEST DEV IN TEST] QWK:{}'
    epoch_msg = epoch_msg.format(va_qwk, te_qwk, tr_log['Epoch_best_dev_qwk'][1], tr_log['Epoch_lowest_dev_loss'][1])
    logger.info(epoch_msg)


def TrainForMultiTraitWithCL(args,
                             Gmodel, Smodel, FCmodel, optims,
                             tr_s_loader, va_s_loader, te_t_loader,
                             t_index, e_index,
                             tr_log, trait_interactive_type, if_directly):
    logger.info('TrainForMultiTrait: [TARGET] P:{} [EPOCH] E:{}...'.format(t_index, e_index))
    if not if_directly and e_index == 1:
        print('分段训练， 现在是初始化模型参数。。。。。。')
        for item_index, (s_item, t_item) in enumerate(zip(tr_s_loader, te_t_loader), start=1):
            Gmodel.train(True)
            Smodel.train(True)
            optims.zero_grad()

            s_prompt, s_pos_ids, s_ling, s_read, s_aes_label = s_item['prompt'], s_item['pos_ids'], s_item['ling'], \
                                                               s_item['read'], s_item['score']
            s_essay_fea = Gmodel(s_pos_ids.to(args.device))
            if trait_interactive_type in ['attention', 'none']:
                s_aes_pre = Smodel(s_essay_fea, s_ling.to(args.device), s_read.to(args.device))
                s_aes_pre = s_aes_pre.to('cpu')
                aes_loss = mask_mse_loss_fn(s_aes_label, s_aes_pre)
                loss = aes_loss
            else:
                s_aes_pre, trait_loss = Smodel(s_essay_fea, s_ling.to(args.device), s_read.to(args.device))
                s_aes_pre = s_aes_pre.to('cpu')
                aes_loss = mask_mse_loss_fn(s_aes_label, s_aes_pre)
                loss = aes_loss + 0.5 * trait_loss
            loss.backward(retain_graph=True)
            optims.step()
    else:
        if if_directly:
            print('不分段训练，直接开始对比学习。。。。。。')
        else:
            print('分段训练，现在开始对比学习。。。。。。')
        s_essay_embed = GetAllEssayRepresentations(args, Gmodel, tr_s_loader)
        t_essay_embed = GetAllEssayRepresentations(args, Gmodel, te_t_loader)
        for item_index, (s_item, t_item) in enumerate(zip(tr_s_loader, te_t_loader), start=1):
            Gmodel.train(True)
            Smodel.train(True)
            FCmodel.train(True)
            optims.zero_grad()

            s_prompt, s_pos_ids, s_ling, s_read, s_aes_label = s_item['prompt'], s_item['pos_ids'], s_item['ling'], s_item['read'], s_item['score']
            t_prompt, t_pos_ids, t_ling, t_read, t_aes_label = t_item['prompt'], t_item['pos_ids'], t_item['ling'], t_item['read'], t_item['score']

            # Start First Step
            s_essay_fea_1 = Gmodel(s_pos_ids.to(args.device))
            t_essay_fea_1 = Gmodel(t_pos_ids.to(args.device))
            cl_loss_1 = 0.5 * FCmodel(s_essay_fea_1, t_essay_fea_1, s_essay_embed.to(args.device), t_essay_embed.to(args.device))
            first_loss = cl_loss_1
            first_loss.backward(retain_graph=True)
            optims.step()
            # End First Step

            s_essay_fea_2 = Gmodel(s_pos_ids.to(args.device))
            t_essay_fea_2 = Gmodel(t_pos_ids.to(args.device))
            cl_loss_2 = 0.5 * FCmodel(s_essay_fea_2, t_essay_fea_2, s_essay_embed.to(args.device), t_essay_embed.to(args.device))

            if trait_interactive_type in ['attention', 'none']:
                s_aes_pre_2 = Smodel(s_essay_fea_2, s_ling.to(args.device), s_read.to(args.device))
                s_aes_pre_2 = s_aes_pre_2.to('cpu')
                aes_loss_2 = mask_mse_loss_fn(s_aes_label, s_aes_pre_2)
                second_loss = aes_loss_2 + cl_loss_2
            else:
                s_aes_pre_2, trait_loss_2 = Smodel(s_essay_fea_2, s_ling.to(args.device), s_read.to(args.device))
                s_aes_pre_2 = s_aes_pre_2.to('cpu')
                aes_loss_2 = mask_mse_loss_fn(s_aes_label, s_aes_pre_2)
                second_loss = aes_loss_2 + cl_loss_2 + 0.5 * trait_loss_2
            second_loss.backward(retain_graph=True)
            optims.step()

            s_essay_fea_3 = Gmodel(s_pos_ids.to(args.device))
            if trait_interactive_type in ['attention', 'none']:
                s_aes_pre_3 = Smodel(s_essay_fea_3, s_ling.to(args.device), s_read.to(args.device))
                s_aes_pre_3 = s_aes_pre_3.to('cpu')
                aes_loss_3 = mask_mse_loss_fn(s_aes_label, s_aes_pre_3)
                third_loss = aes_loss_3
            else:
                s_aes_pre_3, trait_loss_3 = Smodel(s_essay_fea_3, s_ling.to(args.device), s_read.to(args.device))
                s_aes_pre_3 = s_aes_pre_3.to('cpu')
                aes_loss_3 = mask_mse_loss_fn(s_aes_label, s_aes_pre_3)
                third_loss = aes_loss_3 + 0.5 * trait_loss_3
            third_loss.backward(retain_graph=True)
            optims.step()

        va_qwk_set = TestForMultiTrait(args, Gmodel, Smodel, va_s_loader, 'valid', trait_interactive_type)
        te_qwk_set = TestForMultiTrait(args, Gmodel, Smodel, te_t_loader, 'test', trait_interactive_type)
        if va_qwk_set['Avg'] > tr_log['Best_dev_qwk_mean']:
            tr_log['Best_dev_qwk_mean'] = va_qwk_set['Avg']
            tr_log['Best_test_qwk_mean'] = te_qwk_set['Avg']
            tr_log['Best_dev_qwk_set'] = va_qwk_set
            tr_log['Best_test_qwk_set'] = te_qwk_set
            tr_log['Best_epoch'] = e_index
            tr_log['BestModel']['BestGmodel'] = Gmodel.state_dict()
            tr_log['BestModel']['BestSmodel'] = Smodel.state_dict()

        epoch_msg = '[CURRENT TARGET] P: {}  [CURRENT EPOCH] E: {}'.format(t_index, e_index)
        logger.info(epoch_msg)
        for trait in va_qwk_set.keys():
            logger.info('[DEV] {} QWK: {:.2f}'.format(trait, va_qwk_set[trait]))
        logger.info('-' * 20)
        for trait in te_qwk_set.keys():
            logger.info('[TEST] {} QWK: {:.2f}'.format(trait, te_qwk_set[trait]))
        logger.info('-' * 20)
        logger.info('Best Epoch: {}'.format(tr_log['Best_epoch']))
        for trait in tr_log['Best_test_qwk_set'].keys():
            logger.info('[BEST DEV IN TEST] {} QWK: {:.2f}'.format(trait, tr_log['Best_test_qwk_set'][trait]))
        logger.info('-' * 50)


def TrainForMultiTraitWOCL(args,
                             Gmodel, Smodel, optims,
                             tr_s_loader, va_s_loader, te_t_loader,
                             t_index, e_index,
                             tr_log, trait_interactive_type):
    logger.info('TrainForMultiTrait: [TARGET] P:{} [EPOCH] E:{}...'.format(t_index, e_index))
    for item_index, (s_item, t_item) in enumerate(zip(tr_s_loader, te_t_loader), start=1):
        Gmodel.train(True)
        Smodel.train(True)
        optims.zero_grad()

        s_prompt, s_pos_ids, s_ling, s_read, s_aes_label = s_item['prompt'], s_item['pos_ids'], s_item['ling'], \
                                                           s_item['read'], s_item['score']
        s_essay_fea = Gmodel(s_pos_ids.to(args.device))
        if 'attention' in trait_interactive_type or 'none' in trait_interactive_type:
            s_aes_pre = Smodel(s_essay_fea, s_ling.to(args.device), s_read.to(args.device))
            s_aes_pre = s_aes_pre.to('cpu')
            aes_loss = mask_mse_loss_fn(s_aes_label, s_aes_pre)
            loss = aes_loss
        else:
            s_aes_pre, trait_loss = Smodel(s_essay_fea, s_ling.to(args.device), s_read.to(args.device))
            s_aes_pre = s_aes_pre.to('cpu')
            aes_loss = mask_mse_loss_fn(s_aes_label, s_aes_pre)
            loss = aes_loss + 0.5 * trait_loss
        loss.backward(retain_graph=True)
        optims.step()

    va_qwk_set = TestForMultiTrait(args, Gmodel, Smodel, va_s_loader, 'valid', trait_interactive_type)
    te_qwk_set = TestForMultiTrait(args, Gmodel, Smodel, te_t_loader, 'test', trait_interactive_type)
    if va_qwk_set['Avg'] > tr_log['Best_dev_qwk_mean']:
        tr_log['Best_dev_qwk_mean'] = va_qwk_set['Avg']
        tr_log['Best_test_qwk_mean'] = te_qwk_set['Avg']
        tr_log['Best_dev_qwk_set'] = va_qwk_set
        tr_log['Best_test_qwk_set'] = te_qwk_set
        tr_log['Best_epoch'] = e_index
        tr_log['BestModel']['BestGmodel'] = Gmodel.state_dict()
        tr_log['BestModel']['BestSmodel'] = Smodel.state_dict()

    epoch_msg = '[CURRENT TARGET] P: {}  [CURRENT EPOCH] E: {}'.format(t_index, e_index)
    logger.info(epoch_msg)
    for trait in va_qwk_set.keys():
        logger.info('[DEV] {} QWK: {:.2f}'.format(trait, va_qwk_set[trait]))
    logger.info('-' * 20)
    for trait in te_qwk_set.keys():
        logger.info('[TEST] {} QWK: {:.2f}'.format(trait, te_qwk_set[trait]))
    logger.info('-' * 20)
    logger.info('Best Epoch: {}'.format(tr_log['Best_epoch']))
    for trait in tr_log['Best_test_qwk_set'].keys():
        logger.info('[BEST DEV IN TEST] {} QWK: {:.2f}'.format(trait, tr_log['Best_test_qwk_set'][trait]))
    logger.info('-' * 50)

