
import numpy as np
from adabpr import AdaBPR
from functions import *


recommender = 'adabpr'
folder = '../datasets/'
dataset = 'ml_100k_occf'
train_file = folder+dataset+'_training.txt'
test_file = folder+dataset+'_testing.txt'
inf = open('../output/'+recommender+'_results.txt', 'a+')
positions = [1, 5, 10, 15]

if recommender == 'adabpr':
    train_data, Tr, Tr_neg, Te = data_process(train_file, test_file)
    for x in np.arange(-6, -5):
        for y in np.arange(-10, -9):
            cmd_str = 'Dataset:'+dataset+'\n'
            ada_bpr = AdaBPR(num_factors=10, lmbda=10**(-5), theta=2**(-7), num_models=30, metric='MAP', max_iter=30, loss_code=3)
            cmd_str += str(ada_bpr)
            print cmd_str
            ada_bpr.fix_model(train_data, Tr, Tr_neg, Te)
            map_value, auc_value, ndcg, prec, rec = ada_bpr.evaluation(Tr_neg, Te, positions)
            results = 'MAP: %s AUC:%s nDCG:%s ' % (map_value, auc_value, ndcg)
            results += ' '.join(['P@%d:%.6f' % (positions[i], prec[i]) for i in xrange(len(positions))])+' '
            results += ' '.join(['R@%d:%.6f' % (positions[i], rec[i]) for i in xrange(len(positions))])
            inf.write(cmd_str+'\n'+results+'\n')
            print results
