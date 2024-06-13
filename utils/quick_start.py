# coding: utf-8
# @email: enoche.chow@gmail.com

"""
Run application
##########################
"""
from logging import getLogger
from itertools import product
from utils.dataset import RecDataset
from utils.dataloader import TrainDataLoader, EvalDataLoader
from utils.logger import init_logger
from utils.configurator import Config
from utils.utils import init_seed, get_model, get_trainer, dict2str
import platform
import os


def quick_start(model, dataset, config_dict, save_model=True, mg=False):
    # merge config dict
    config = Config(model, dataset, config_dict, mg)
    init_logger(config)
    logger = getLogger()
    # print config infor
    logger.info('██Server: \t' + platform.node())
    logger.info('██Dir: \t' + os.getcwd() + '\n')
    logger.info(config)

    # load data
    dataset = RecDataset(config)
    # print dataset statistics
    logger.info(str(dataset))

    train_dataset, valid_dataset, test_dataset = dataset.split()
    logger.info('\n====Training====\n' + str(train_dataset))
    logger.info('\n====Validation====\n' + str(valid_dataset))
    logger.info('\n====Testing====\n' + str(test_dataset))

    # wrap into dataloader
    train_data = TrainDataLoader(config, train_dataset, batch_size=config['train_batch_size'], shuffle=True)
    (valid_data, test_data) = (
        EvalDataLoader(config, valid_dataset, additional_dataset=train_dataset, batch_size=config['eval_batch_size']),
        EvalDataLoader(config, test_dataset, additional_dataset=train_dataset, batch_size=config['eval_batch_size']))

    ############ Dataset loadded, run model
    hyper_ret = []
    val_metric = config['valid_metric'].lower()
    best_test_value = 0.0
    idx = best_test_idx = 0

    logger.info('\n\n=================================\n\n')

    # hyper-parameters
    hyper_ls = []
    if "seed" not in config['hyper_parameters']:
        config['hyper_parameters'] = ['seed'] + config['hyper_parameters']
    for i in config['hyper_parameters']:
        hyper_ls.append(config[i] or [None])
    # combinations
    combinators = list(product(*hyper_ls))
    total_loops = len(combinators)

    hyper_tuple = (42, 'add', 0.001, 0.0001) #DRAGON #論文最佳參數 #["aggr_mode", "reg_weight", "learning_rate"]
    #hyper_tuple = (42, 'add', 0.001, 0.00003) #DRAGON #論文最佳參數-改 #["aggr_mode", "reg_weight", "learning_rate"] 太久 9hr up
    # 記得刪C:\Users\jeffy\python\MMRec-master\data\clothing內的mm_adj_10.pt(項目同質圖)
    #hyper_tuple = ('add', 0.0001, 0.0001, 999) #DRAGON
    hyper_tuple = (999, 0.9, 0.001) #FREEDOM
    hyper_tuple =  (42, 3, 0.001, 0.5) #BM3 ["n_layers", "reg_weight", "dropout"] 論文較佳-大datasets
    hyper_tuple =  (42, 0.0001) # VBPR
    hyper_tuple =  (42, 0.0001, 0.001) # GRCN
    hyper_tuple =  (42, 0.0001, 0.001) # MMGCN
    #hyper_tuple =  (42, 0.01) #MGCN


    for hyper_tuple in combinators:
        # random seed reset
        for j, k in zip(config['hyper_parameters'], hyper_tuple):
            config[j] = k
        init_seed(config['seed'])

        logger.info('========={}/{}: Parameters:{}={}======='.format(
            idx+1, total_loops, config['hyper_parameters'], hyper_tuple))

        # set random state of dataloader
        train_data.pretrain_setup()
        # model loading and initialization
        model = get_model(config['model'])(config, train_data).to(config['device'])
        logger.info(model)

        # trainer loading and initialization
        trainer = get_trainer()(config, model, mg)
        # debug
        # model training
        best_valid_score, best_valid_result, best_test_upon_valid = trainer.fit(
            train_data, valid_data=valid_data, test_data=test_data, saved=save_model)
        
        import torch
        torch.save(model, 'C:/Users/jeffy/python/MMRec-master_new/save/VBPR.pt')

        #########
        hyper_ret.append((hyper_tuple, best_valid_result, best_test_upon_valid))

        # save best test
        if best_test_upon_valid[val_metric] > best_test_value:
            best_test_value = best_test_upon_valid[val_metric]
            best_test_idx = idx
        idx += 1

        logger.info('best valid result: {}'.format(dict2str(best_valid_result)))
        logger.info('test result: {}'.format(dict2str(best_test_upon_valid)))
        logger.info('████Current BEST████:\nParameters: {}={},\n'
                    'Valid: {},\nTest: {}\n\n\n'.format(config['hyper_parameters'],
            hyper_ret[best_test_idx][0], dict2str(hyper_ret[best_test_idx][1]), dict2str(hyper_ret[best_test_idx][2])))

    # log info
    logger.info('\n============All Over=====================')
    for (p, k, v) in hyper_ret:
        logger.info('Parameters: {}={},\n best valid: {},\n best test: {}'.format(config['hyper_parameters'],
                                                                                  p, dict2str(k), dict2str(v)))

    logger.info('\n\n█████████████ BEST ████████████████')
    logger.info('\tParameters: {}={},\nValid: {},\nTest: {}\n\n'.format(config['hyper_parameters'],
                                                                   hyper_ret[best_test_idx][0],
                                                                   dict2str(hyper_ret[best_test_idx][1]),
                                                                   dict2str(hyper_ret[best_test_idx][2])))



import pandas as pd
import numpy as np
from sklearn.metrics import ndcg_score
from tqdm import tqdm

import torch
from utils.dataset import RecDataset
from utils.dataloader import TrainDataLoader, EvalDataLoader
from utils.logger import init_logger
from utils.configurator import Config
from utils.utils import init_seed, get_model, get_trainer, dict2str

import os
os.chdir("C:/Users/jeffy/python/MMRec-master_new/src")

# Load interaction data
inter = pd.read_csv('C:/Users/jeffy/python/MMRec-master_new/data/clothing/clothing.inter', sep='\t')
model = torch.load('C:/Users/jeffy/python/MMRec-master_new/save/dragon_interupt.pt')


torch.save(model.state_dict(), 'C:/Users/jeffy/python/MMRec-master_new/save/dragon_interupt_state_dict.pt')

res = model.full_sort_predict_df([37059])


df = pd.DataFrame()

for temp_cid in inter[inter['cid'].str[:2]=='sf']['cid'].unique():
    #temp_cid = 'sf000001'
    # 從數據中選取survey測試用戶的數據
    temp_userID = inter.loc[inter['cid']==temp_cid, 'userID'].unique()[0]
    temp_test = inter[(inter['userID']==temp_userID) & (inter['x_label']!=0)]

    # 使用模型進行預測
    #res = model.full_sort_predict_df([temp_userID])

    predict_list = model.full_sort_predict([temp_userID]).tolist()
    res = pd.DataFrame({
            'itemID':list(range(len(predict_list))),
            'score':predict_list,
            }).sort_values("score", ascending=False)

    # 將預測結果與測試數據進行合併，並按照得分進行排序
    temp_test= pd.merge(temp_test, res, how='left').sort_values("score", ascending=False).reset_index(drop=True)
    #temp_test = temp_test[(temp_test['weight'] == 50) | (temp_test['weight'] == -20)].reset_index(drop=True)
    if(len(temp_test)==0):
        continue
    # 定義計算NDCG的函數
    def calculate_ndcg(df, rank, relevance_col):
        df_sorted = df.sort_values(by='score', ascending=False)
        true_relevance = np.asarray([df_sorted[relevance_col].values])
        scores = np.asarray([df_sorted['score'].values])
        return ndcg_score(true_relevance, scores, k=rank)
    
    
    def calculate_recall(df, rank, relevance_col):
        # 按照得分對數據進行排序
        df_sorted = df.sort_values(by='score', ascending=False)
        # 獲取前rank個推薦的相關性
        top_relevance = df_sorted[relevance_col].values[:rank]
        # 計算並返回Recall
        return sum(top_relevance) / len(df[df[relevance_col] == 1])

    # 將x_label轉換為二元相關性：如果x_label為1或2，則相關性為1，否則為0
    temp_test['relevance'] = temp_test['weight'].apply(lambda x: 1 if x>0 else 0)
    #temp_test['relevance']

    # 只取top sku，去除view影響
    temp_test['top_sku'] = temp_test['sku'].apply(lambda x: x.split(' ')[0])
    temp_test = temp_test.drop_duplicates('top_sku').reset_index(drop=True)
    
    res_df = temp_test.loc[:,['sku', 'top_sku', 'relevance', 'score']]

    #calculate_ndcg(temp_test, 5, 'relevance')
    #calculate_ndcg(temp_test, 10, 'relevance')

    # 計算並打印Recall@5
    #calculate_recall(temp_test, 5, 'relevance')
    #calculate_recall(temp_test, 10, 'relevance')
    # temp_test.to_excel("sf000001.xlsx")

    # 假設relevance=1表示商品是用戶感興趣的
    relevant_items = temp_test[:5][temp_test['relevance'] == 1]
    precision_at_5 = len(relevant_items) / 5

    tmp = pd.DataFrame({
        'cid':temp_cid,
        #'recall@5':calculate_recall(temp_test, 5, 'relevance'),
        'recall@10':calculate_recall(temp_test, 10, 'relevance'),
        'ndcg@5':calculate_ndcg(temp_test, 5, 'relevance'),
        'ndcg@10':calculate_ndcg(temp_test, 10, 'relevance'),
        'precision@5':precision_at_5,
        'like_num':sum(temp_test['relevance'])
    }, index=[0])

    df = pd.concat([df, tmp])

    average_values = df.drop(columns='cid').mean()

df = df.sort_values('cid').reset_index(drop=True)
print(df)
print(average_values)

import numpy as np
# Load test IDs and process with model
test_userID = np.load('C:/Users/jeffy/python/MMRec-master_new/data/clothing/test_userID.npy')
test_userID = np.load('C:/Users/jeffy/python/MMRec-master_new/data/clothing/vip_userID.npy')
test_itemID = np.load('C:/Users/jeffy/python/MMRec-master_new/data/clothing/test_itemID.npy')
#model.full_sort_predict_df([0])

def calculate_ndcg(df_sorted, rank, relevance_col):
    true_relevance = np.asarray([df_sorted[relevance_col].values[:rank]])
    scores = np.asarray([df_sorted['score'].values[:rank]])
    return ndcg_score(true_relevance, scores, k=rank)

def calculate_recall(df_sorted, rank, relevance_col):
    top_relevance = df_sorted[relevance_col].values[:rank]
    total_relevant_items = sum(df_sorted[relevance_col] == 1)
    if total_relevant_items == 0:
        return 0     
    return sum(top_relevance) / total_relevant_items


# 提前對 inter 進行分組
grouped_inter_test = inter[inter['x_label'] != 0].groupby('userID')
grouped_inter_train = inter[inter['x_label'] == 0].groupby('userID')

results = []
from tqdm import tqdm

k_num=10
# temp_userID=482
j=0
for temp_userID in tqdm(test_userID):
    j=j+1
    #if(j>100):
    #    break
    if temp_userID not in grouped_inter_test.groups:
        continue
    try:
        temp_test = grouped_inter_test.get_group(temp_userID)
        temp_train = grouped_inter_train.get_group(temp_userID)
    
    except:
        continue
    # 使用模型進行預測
    res = model.full_sort_predict_df([temp_userID])

    # remove items in train set
    res = res[~res['itemID'].isin(temp_train['itemID'])]

    # 將預測結果與測試數據進行合併，並按照得分進行排序
    res = res[res['itemID'].isin(test_itemID)]

    # 使用merge合併weight到res中
    merged = pd.merge(res, temp_test[['itemID', 'weight']], on='itemID', how='left').dropna()
    if(len(merged)<3):
        continue
    #merged = res.merge(temp_test[['itemID', 'weight']], on='itemID', how='left').dropna()

    # 使用fillna填充NaN值，因為不是所有的itemID都在temp_test中
    #merged['weight'].fillna(0, inplace=True)
    merged['weight'] = merged['weight']/50
    
    # 假設 model.full_sort_predict_df 返回的 DataFrame 是按照 'score' 降序排序的
    merged['relevance'] = 0
    merged.loc[merged['weight']>=0.2, 'relevance'] = 1

    #merged[merged['relevance']==1]

    
    # 從已排序的 DataFrame 中計算各種評價指標
    recall = calculate_recall(merged, k_num, 'relevance')
    ndcg = calculate_ndcg(merged, k_num, 'weight')
    
    precision = sum(i > 0 for i in merged['relevance'].values[:k_num])/k_num
        
    results.append({
        'cid': temp_userID,
        'recall@10': recall,
        'ndcg@10': ndcg,
        'precision@10': precision
    })

df = pd.DataFrame(results)
average_values = df.drop(columns='cid').mean()
average_values