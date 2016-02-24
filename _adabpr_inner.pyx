
import time
import random
import numpy as np
cimport numpy as np
np.import_array()


DEF MAX_USER_NUM = 100000
DEF MAX_ITEM_NUM = 100000

cdef extern from "adabpr_inner.h":
    long int uniform_random_id(long int *list, int size)
    
    double mean_average_precision_fast(double *pos_val, double *neg_val, long int num_pos, long int num_neg)

    double adabpr_auc_fast_loss(double *U_pt, double *V_pt, double *W_pt, 
        long int *pair_data, long int num_pairs, int loss_code, int num_factors)
    
    void adabpr_auc_fast_train(double *U_pt, double *V_pt, long int *train_pt, double *train_wpt, long int num_train, long int *valid_pt, double *valid_wpt, long int num_valid, long int **neg_items, int *neg_size, int num_factors, double theta, double reg_u, double reg_i, int loss_code, int max_iter)

    void adabpr_map_fast_update(double *U_pt, double *V_pt, long int *train_pt, double *train_wpt, long int num_train, long int **neg_items, int *neg_size, int num_factors, int gamma, double curr_theta, double reg_u, double reg_i)


def mean_average_precision(pos_inx, neg_inx, val):
    pos_val, neg_val = val[pos_inx], val[neg_inx]
    ii = np.argsort(pos_val)[::-1]
    jj = np.argsort(neg_val)[::-1]
    num_pos, num_neg = len(pos_inx), len(neg_inx)
    pos_sort, neg_sort = pos_val[ii], neg_val[jj]
    return mean_average_precision_fast(<double *>np.PyArray_DATA(pos_sort), <double *>np.PyArray_DATA(neg_sort), num_pos, num_neg)


def sample_validation_triples(index_dict, Tr, Tr_neg, ratio):
    valid_data = []
    num = max(int(np.ceil(ratio*len(Tr.keys()))), 100)
    sub_set = random.sample(Tr.keys(), num)
    for u in sub_set:
        valid_data.extend([(u, i, uniform_random_id(<long int*>np.PyArray_DATA(
            Tr_neg[u]["items"]), Tr_neg[u]["num"])) for i in Tr[u]['items']])
    inx = [index_dict[u][i] for u, i, j in valid_data]
    return np.array(valid_data), np.array(inx)


def adabpr_auc_train(model, ccf_inx, train_data, Tr, Tr_neg):
    cdef double *U_pt = <double *>(np.PyArray_DATA(model.U[ccf_inx]))
    cdef double *V_pt = <double *>(np.PyArray_DATA(model.V[ccf_inx]))
    cdef double *W_pt = <double *>(np.PyArray_DATA(model.D))
    cdef long int *train_pt = <long int *>(np.PyArray_DATA(train_data))
    cdef int neg_size[MAX_USER_NUM]
    cdef long int *neg_items[MAX_USER_NUM]
    for u in xrange(model.num_users):
        neg_size[u] = <int> Tr_neg[u]["num"]
        neg_items[u] = <long int *> np.PyArray_DATA(Tr_neg[u]["items"])
    # optimize the weighted auc
    if model.metric == 'AUC':
        valid_data, ii = sample_validation_triples(model.index_dict, Tr, Tr_neg, 0.1)
        valid_pt = <long int*>np.PyArray_DATA(valid_data) 
        valid_wpt = <double *>np.PyArray_DATA(model.D[ii])
        num_valid = valid_data.shape[0]
        adabpr_auc_fast_train(U_pt, V_pt, train_pt, W_pt, model.num_train, valid_pt, valid_wpt, num_valid, neg_items, neg_size, model.d, model.theta, model.lmbda, model.lmbda, model.loss_code, model.max_iter)
    
    # optimize the weighted map
    elif model.metric == 'MAP':
        UK, VK = model.U[ccf_inx].copy(), model.V[ccf_inx].copy()
        num_valid_users = max(int(np.floor(model.num_users*1)), 100)
        valid_users = np.random.choice(model.num_users, num_valid_users, replace=False)
        curr_theta, obj = model.theta, 0.0
        for t in xrange(model.max_iter):
            adabpr_map_fast_update(U_pt, V_pt, train_pt, W_pt, model.num_train, neg_items, neg_size, model.d, model.gamma, curr_theta, model.lmbda, model.lmbda)
            curr_theta *= 0.9
            predict_val = np.dot(model.U[ccf_inx][valid_users, :], model.V[ccf_inx].T)
            train_map=sum([mean_average_precision(Tr[valid_users[i]]['items'], Tr_neg[valid_users[i]]['items'], predict_val[i]) for i in xrange(num_valid_users)])/num_valid_users
            # print train_map
            if train_map>obj:
                obj = train_map
                UK, VK = model.U[ccf_inx].copy(), model.V[ccf_inx].copy()
            else:
                model.U[ccf_inx], model.V[ccf_inx] = UK, VK
                break
