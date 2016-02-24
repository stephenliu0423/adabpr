
long int uniform_random_id(long int *list, int size);

double mean_average_precision_fast(double *pos_val, double *neg_val, long int num_pos, long int num_neg);

double adabpr_auc_fast_loss(double *U_pt, double *V_pt, double *W_pt, long int *pair_data, long int num_pairs, int loss_code, int num_factors);

void adabpr_auc_fast_train(double *U_pt, double *V_pt, long int *train_pt, double *train_wpt, long int num_train, long int *valid_pt, double *valid_wpt, long int num_valid, long int **neg_items, int *neg_size, int num_factors, double theta, double reg_u, double reg_i, int loss_code, int max_iter);

void adabpr_map_fast_update(double *U_pt, double *V_pt, long int *train_pt, double *train_wpt, long int num_train, long int **neg_items, int *neg_size, int num_factors, int gamma, double curr_theta, double reg_u, double reg_i);