
#define min(x, y) (((x) < (y)) ? (x) : (y))
#define max(x, y) (((x) > (y)) ? (x) : (y))

#include <stdio.h>
#include <malloc.h>
#include <math.h>
#include <stdlib.h>


void compute_auc_list_func(long int *pos, long int *neg, double *val, double *measure, long int num_pos, long int num_neg){
    
    long int i, j;
    double num;
    for(i=0; i<num_pos; i++){
        num = 0.0;
        for(j=0; j<num_neg; j++){
            if(val[pos[i]]>val[neg[j]])
                num+=1.0;
            // printf("%g,%g\n", val[pos[i]], val[neg[j]]);
        }
        measure[i]=num/num_neg;
    }
}

void compute_auc_at_N_list_func(long int *pos, long int *neg, double *val, double *measure, long int num_pos, long int num_neg, int N){
    
    long int i, j;
    double num;
    for(i=0; i<num_pos; i++){
        num = 0.0;
        for(j=0; j<num_neg; j++){
            if(val[pos[i]]<=val[neg[j]])
                num+=1.0;
            // printf("%g,%g\n", val[pos[i]], val[neg[j]]);
        }
        if(num>=N)
            measure[i]=0.0;
        else
            measure[i]=1.0 - num/N;
    }
}


void compute_map_list_func(long int *pos, long int *neg, double *val, double *measure, long int num_pos, long int num_neg){
    
    long int i, j;
    double num;
    for(i=0; i<num_pos; i++){
        num = 0.0;
        for(j=0; j<num_neg; j++){
            if(val[pos[i]]<=val[neg[j]])
                num += 1.0;
            // printf("%g,%g\n", val[pos[i]], val[neg[j]]);
        }
        measure[i]=(i+1)/(i+num+1);
    }
}

long int auc_computation_func(long int *pos, long int *neg, double *val, long int num_pos, long int num_neg){
    long int i, j, num=0;
    for(i=0; i<num_pos; i++){
        for(j=0; j<num_neg; j++){
            if(val[pos[i]]>val[neg[j]])
                num++;
        }
    }
    return num;
}

long int uniform_random_id(long int *list, int size){
    long int i;
    i = (rand() % size);
    return list[i];
}

double mean_average_precision_fast(double *pos_val, double *neg_val, long int num_pos, long int num_neg){
    long int i, j;
    double num, map=0.0;

    for(i=0; i<num_pos; i++){
        num = 0.0;
        for(j=0; j<num_neg;j++){
            if(pos_val[i]<=neg_val[j])
                num += 1.0;
        }
        map += (i+1)/(i+num+1);
    }

    return (map/num_pos);
}


double adabpr_auc_fast_loss(double *U_pt, double *V_pt, double *W_pt, long int *pair_data, long int num_pairs, int loss_code, int num_factors){

    double val, w, loss=0;
    long int uid, pid1, pid2, k; int d;

    for(k=0; k<num_pairs; k++){
        uid=pair_data[k*3]; pid1=pair_data[k*3+1]; pid2=pair_data[k*3+2]; val=0;
        for(d=0; d<num_factors; d++)
            val += U_pt[uid*num_factors+d]*(V_pt[pid2*num_factors+d]-V_pt[pid1*num_factors+d]);
        w=W_pt[k];
        if (loss_code==1)
            loss += w*log(1.0+exp(val));
        else if (loss_code==2)
            loss += -w*(1.0/(1.0+exp(val)));
        else if (loss_code==3){
            val += 1.0;
            loss += w*((val>0.0)? val : 0.0);
        }
        else if(loss_code==4){
            loss += w*exp(val);
        }
        else if(loss_code==5){
            loss += 0.5*w*pow((1.0+val), 2);
        }
    }
    return loss;
}


void adabpr_auc_fast_train(double *U_pt, double *V_pt, long int *train_pt, double *train_wpt, long int num_train, long int *valid_pt, double *valid_wpt, long int num_valid, long int **neg_items, int *neg_size, int num_factors, double theta, double reg_u, double reg_i, int loss_code, int max_iter){

	int d, t;
	long int uid, pid1, pid2, k, k1;
	double val, uk, vk1, vk2, w, curr_theta, last_loss, curr_loss, delta_loss; 

	curr_theta = theta;
	last_loss = adabpr_auc_fast_loss(U_pt, V_pt, valid_wpt, valid_pt, num_valid, loss_code, num_factors);

	for(t=0; t<max_iter; t++){
		for(k=0; k<num_train; k++){
			uid = train_pt[k*2];
			pid1 = train_pt[k*2+1];
			w = train_wpt[k];
			k1 = (rand() % neg_size[uid]);
			pid2 = neg_items[uid][k1];
			val = 0.0;
			for(d=0; d<num_factors; d++)
		        val += U_pt[uid*num_factors+d]*(V_pt[pid1*num_factors+d]-V_pt[pid2*num_factors+d]);
		    if (loss_code == 1)
		        val = -1.0/(1.0+exp(val));
		    else if(loss_code==2){
		        val = exp(val);
		        val = -val/pow((1+val), 2);
		    }
		    else if (loss_code==3){
		        val = (val<1.0) ? -1.0 : 0.0;
		    }
		    else if(loss_code==4){
		        val = -exp(-val);
		    }
		    else if(loss_code==5){
		        val =val-1.0;
		    }
		    for(d=0; d<num_factors; d++){
		        uk=U_pt[uid*num_factors+d]; 
		        vk1=V_pt[pid1*num_factors+d]; 
		        vk2=V_pt[pid2*num_factors+d];
		        U_pt[uid*num_factors+d] -= curr_theta*(val*w*(vk1-vk2)+reg_u*uk);
		        V_pt[pid1*num_factors+d] -= curr_theta*(val*w*uk+reg_i*vk1);
		        V_pt[pid2*num_factors+d] -= curr_theta*(-val*w*uk+reg_i*vk2);
		    }
		}
		curr_loss = adabpr_auc_fast_loss(U_pt, V_pt, valid_wpt, valid_pt, num_valid, loss_code, num_factors);

		delta_loss = (curr_loss-last_loss)/fabs(last_loss);
        if (fabs(delta_loss) < 1e-5)
        	break;
        curr_theta = 0.9*curr_theta;
        last_loss = curr_loss;
	}
}


double warp_compute_rank_loss(float num, float t){
    int i; double loss=0;
    int rank =(int) floor((num-1)/t);
    for(i=1; i<rank+1; i++)
        loss += 1.0/i;
    return loss;
}


void adabpr_map_fast_update(double *U_pt, double *V_pt, long int *train_pt, double *train_wpt, long int num_train, long int **neg_items, int *neg_size, int num_factors, int gamma, double curr_theta, double reg_u, double reg_i){

    long int uid, pid1, pid2, k, k1;
    double w, val1, val2, loss, uk, vk1, vk2;
    int d, t; int flag;

    for(k=0; k<num_train; k++){
        uid = train_pt[2*k]; pid1 = train_pt[2*k+1]; w=train_wpt[k];
        val1=0;
        for(d=0; d<num_factors; d++)
            val1 += U_pt[uid*num_factors+d]*V_pt[pid1*num_factors+d];
        t = 0; flag = 0;
        while(t<neg_size[uid]/gamma){
            t++;
            k1 = (rand() % neg_size[uid]);
            pid2 = neg_items[uid][k1];            
            val2 = 0;
            for(d=0; d<num_factors; d++)
                val2 += U_pt[uid*num_factors+d]*V_pt[pid2*num_factors+d];
            if(val1<val2+1){
                flag=1;
                break;
            }
        }
        // update the model parameters
        if(flag){
            loss = warp_compute_rank_loss((float) neg_size[uid], (float) t);
            // loss = 1.0;
            for(d=0; d<num_factors; d++){
                uk = w*loss*(V_pt[pid2*num_factors+d]-V_pt[pid1*num_factors+d])+reg_u*U_pt[uid*num_factors+d];
                vk1 = -w*loss*U_pt[uid*num_factors+d] + reg_i*V_pt[pid1*num_factors+d];
                vk2 = w*loss*U_pt[uid*num_factors+d] + reg_i*V_pt[pid2*num_factors+d];
                U_pt[uid*num_factors+d] -= curr_theta*uk;
                V_pt[pid1*num_factors+d] -= curr_theta*vk1;
                V_pt[pid2*num_factors+d] -= curr_theta*vk2;
            }
        }
        else{
            for(d=0; d<num_factors; d++){
                uk = reg_u*U_pt[uid*num_factors+d];
                vk1 = reg_i*V_pt[pid1*num_factors+d];
                U_pt[uid*num_factors+d] -= curr_theta*uk;
                V_pt[pid1*num_factors+d] -= curr_theta*vk1;
            }
        }
    }
}