
import pdb
import time
import numpy as np
from collections import defaultdict
from evaluation import precision, recall, nDCG
from adabpr_inner import adabpr_auc_train, compute_auc_list, compute_map_list, compute_auc_at_N_list, mean_average_precision, auc_computation


class AdaBPR:

    def __init__(self, num_factors=100, lmbda=0.015, theta=0.05, num_models=3,
                 max_iter=50, metric='auc', loss_code=1, gamma=1):
        self.d = num_factors
        self.lmbda = lmbda
        self.theta = theta
        self.num_models = num_models
        self.max_iter = max_iter
        self.metric = metric
        self.gamma = gamma
        self.loss_code = loss_code  # 1 for log-loss, 2 for sigmoid loss, 3 for hinge loss, 4 for exponential loss

    def update_sample_weights(self, Tr, Tr_neg):
        measure = self.compute_rank_loss(self.M, Tr, Tr_neg)
        val = np.mean(measure)
        v = np.exp(-measure)*self.D0
        self.D = (v/np.sum(v))*self.num_train
        return val

    def compute_rank_loss(self, predict_val, Tr, Tr_neg, N=10):
        accuracy = np.zeros(self.num_train)
        if self.metric == 'AUC':
            for u in xrange(self.num_users):
                pos_inx, num_pos = Tr[u]['items'], Tr[u]['num']
                measure = np.zeros(num_pos)
                compute_auc_list(pos_inx, Tr_neg[u]["items"], predict_val[u], measure)
                for i, v in enumerate(measure):
                    ii = self.index_dict[u][pos_inx[i]]
                    accuracy[ii] = v
        elif self.metric == 'MAP':
            for u in xrange(self.num_users):
                pos_inx, num_pos = Tr[u]['items'], Tr[u]['num']
                neg_inx, num_neg = Tr_neg[u]['items'], Tr_neg[u]['num']
                ii = np.argsort(predict_val[u][pos_inx])[::-1]
                jj = np.argsort(predict_val[u][neg_inx])[::-1]
                measure = np.zeros(num_pos)
                pos_sort, neg_sort = pos_inx[ii], neg_inx[jj]
                compute_map_list(pos_sort, neg_sort, predict_val[u], measure, num_pos, num_neg)
                for i, v in enumerate(measure):
                    kk = self.index_dict[u][pos_sort[i]]
                    accuracy[kk] = v
        elif self.metric == 'AUC@N':
            for u in xrange(self.num_users):
                pos_inx, num_pos = Tr[u]['items'], Tr[u]['num']
                measure = np.zeros(num_pos)
                compute_auc_at_N_list(pos_inx, Tr_neg[u]["items"], predict_val[u], measure, N)
                for i, v in enumerate(measure):
                    ii = self.index_dict[u][pos_inx[i]]
                    accuracy[ii] = v
        elif self.metric == 'nDCG':
            pass
        return accuracy

    def update_recommender_weights(self, index, Tr, Tr_neg):
        predict_val = self.U[index].dot(self.V[index].T)
        measure = self.compute_rank_loss(predict_val, Tr, Tr_neg)
        self.W[index] = 0.5*np.log(np.sum(self.D*(1+measure))/np.sum(
            self.D*(1-measure)))
        self.M += self.W[index]*predict_val

    def init_sample_weights(self, train_data, Tr):
        self.D0 = np.array([1.0/Tr[u]['num'] for u, i in train_data])
        self.D = (self.D0/np.sum(self.D0))*self.num_train
        self.index_dict = defaultdict(lambda: defaultdict(int))
        for inx in xrange(train_data.shape[0]):
            u, i = train_data[inx]
            self.index_dict[u][i] = inx

    def fix_model(self, train_data, Tr, Tr_neg, Te):
        self.num_train = train_data.shape[0]
        self.num_users = np.max(train_data[:, 0])+1
        self.num_items = np.max(train_data[:, 1])+1
        prng = np.random.RandomState(seed=100)
        self.U = [prng.rand(self.num_users, self.d) for i in xrange(self.num_models)]
        self.V = [prng.rand(self.num_items, self.d) for i in xrange(self.num_models)]

        # self.U = [np.sqrt(1/float(self.d))*np.random.randn(self.num_users, self.d) for i in xrange(self.num_models)]
        # self.V = [np.sqrt(1/float(self.d))*np.random.randn(self.num_items, self.d) for i in xrange(self.num_models)]

        self.W = np.zeros(self.num_models)  # weights for component recommenders
        self.M = np.zeros((self.num_users, self.num_items))
        self.init_sample_weights(train_data, Tr)
        for index in xrange(self.num_models):
            t1 = time.clock()
            adabpr_auc_train(self, index, train_data, Tr, Tr_neg)
            self.update_recommender_weights(index, Tr, Tr_neg)
            if self.num_models > 1:
                val = self.update_sample_weights(Tr, Tr_neg)
                # print 'the %s th ccf model, valid_measure: %.6f time used:%.6f' % (
                    # index+1, val, time.clock()-t1)

    def evaluation(self, Tr_neg, Te, positions=[5, 10, 15]):
        prec = np.zeros(len(positions))
        rec = np.zeros(len(positions))
        map_value, auc_value, ndcg = 0.0, 0.0, 0.0
        for u in Te:
            val = self.M[u, :]
            inx = Tr_neg[u]['items']
            A = set(Te[u])
            B = set(inx) - A
            # compute precision and recall
            ii = np.argsort(val[inx])[::-1][:max(positions)]
            prec += precision(Te[u], inx[ii], positions)
            rec += recall(Te[u], inx[ii], positions)
            ndcg_user = nDCG(Te[u], inx[ii], 10)
            # compute map and AUC
            pos_inx = np.array(list(A))
            neg_inx = np.array(list(B))
            map_user = mean_average_precision(pos_inx, neg_inx, val)
            auc_user = auc_computation(pos_inx, neg_inx, val)
            ndcg += ndcg_user
            map_value += map_user
            auc_value += auc_user
            # outf.write(" ".join([str(map_user), str(auc_user), str(ndcg_user)])+"\n")
        # outf.close()
        return map_value/len(Te.keys()), auc_value/len(Te.keys()), ndcg/len(Te.keys()), prec/len(Te.keys()), rec/len(Te.keys())

    def __str__(self):
        return "Recommender: AdaBPR, num_factors:%s, lmbda:%s, theta:%s, num_models:%s, metric:%s, loss_code:%s, max_iter:%s" % (self.d, self.lmbda, self.theta, self.num_models, self.metric, self.loss_code, self.max_iter)
