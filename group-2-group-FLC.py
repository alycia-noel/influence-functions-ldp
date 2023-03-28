import time
import math
import copy
import torch
import pickle
import random
import logging
import warnings
import datetime
import pandas as pd
import numpy as np
import seaborn as sns
import torch.nn as nn
import torch.optim as opt
import matplotlib.pyplot as plt

from tqdm import tqdm
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from torch.autograd import grad
from torch.autograd import Variable
from torch.autograd.functional import vhp
from get_datasets import get_diabetes, get_adult, get_law
from torch.utils.data import Subset, DataLoader
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")

def visualize_result(e_k_actual, e_k_estimated, ep, title):

    fig, ax = plt.subplots()
    palette = sns.color_palette("husl", len(e_k_actual))
  
    actual = []
    estimated = []

    actual_all = []
    estimated_all = []
    spearman_all = []
    mae_all = []
    
    for x in range(len(e_k_actual)):
        actual_all.extend(e_k_actual[x])
        estimated_all.extend(e_k_estimated[x])
    
    for i in range(len(e_k_actual)):
        spear = spearmanr(e_k_actual[i], e_k_estimated[i]).correlation
        if not math.isnan(spear):
            spearman_all.append(spear)
        mae_all.append(mean_absolute_error(e_k_actual[i], e_k_estimated[i]))
        
    max_abs = np.max([np.abs(actual_all), np.abs(estimated_all)])
    min_, max_ = -max_abs * 1.1, max_abs * 1.1
    
    plt.rcParams['figure.figsize'] = 6, 5
    
    for k in range(len(e_k_actual)):
        ax.scatter(e_k_actual[k], e_k_estimated[k], zorder=2, s=10, color = palette[k], label=ep[k])

    ax.set_title(f'Actual vs. Estimated loss')
    ax.set_xlabel('Actual loss diff')
    ax.set_ylabel('Estimated loss diff')
   
    ax.set_xlim(min(actual_all)-.0005, max(actual_all)+.0005)
    ax.set_ylim(max(estimated_all)+.0005,min(estimated_all)-.0005)
    
    z = np.polyfit(actual_all, estimated_all, 1)
    p = np.poly1d(z)

    #add trendline to plot
    ax.plot(actual_all, p(actual_all), ls="-")
    text = 'MAE = {:.03}\nP = {:.03}'.format(sum(mae_all)/len(e_k_actual), sum(spearman_all)/len(e_k_actual))
    ax.text(max(actual_all)-.0005, max(estimated_all)-.0005, text, verticalalignment='bottom', horizontalalignment='right')
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.show()
    
class CreateData(torch.utils.data.Dataset):
    def __init__(self, data, targets, pert_status):
        self.data = data
        self.targets = targets
        self.pert = pert_status

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        out_data = self.data[idx]
        out_label = self.targets[idx]
        pert_label = self.pert[idx]

        return out_data, out_label, pert_label
    
def get_data(new_train_df, feature_set, label, k):    

    # based on Race or Gender
    selected_group = new_train_df.loc[new_train_df['sex'] == 0]

    num_to_sample = round((k / 100)*len(selected_group))

    sampled_group = selected_group.sample(n=num_to_sample)
    not_selected = new_train_df.drop(sampled_group.index)

    selected_group_X = sampled_group[feature_set]
    selected_group_y = sampled_group[label]

    not_selected_group_X = not_selected[feature_set]
    not_selected_group_y = not_selected[label]   
   
    return selected_group_X, selected_group_y, not_selected_group_X, not_selected_group_y

def randomize_resp(label, epsilon):

    probability = float(math.e ** epsilon) / float(1 + (math.e ** epsilon))
    
    if label == 0:
        new_label = np.random.choice([0,1], p=[probability, 1-probability])
    else:
        new_label = np.random.choice([0,1], p=[1-probability, probability])

    return new_label

def get_p(epsilon):
    probability = float(math.e ** epsilon) / float(1 + (math.e ** epsilon))
    p = torch.FloatTensor([[probability, 1-probability], [1-probability, probability]])
    
    return p

def to_categorical(y, num_classes, act_or_pred):
    input_shape = y.shape

    categorical = []
    
    if act_or_pred == 'pred':
        probabilities = torch.sigmoid(y)
        
        probs_2d = torch.zeros(len(probabilities), 2)
        
        for i, p in enumerate(probabilities):
            probs_2d[i][0] = p
            probs_2d[i][1] = 1-p
            
        return probs_2d
    else:
        for i in range(len(y)):
            if y[i] < 0.5:
                categorical.append([1, 0])
            else:
                categorical.append([0, 1])

        categorical = torch.FloatTensor(categorical)    

        output_shape = input_shape + (num_classes,)
        categorical = torch.reshape(categorical, output_shape)
    
    return categorical

def forward_correct_loss(y_actual, y_pred, epsilon, criterion):
    p = get_p(epsilon)
    
    y_actual_c = to_categorical(y_actual, 2, 'orig')
    y_pred_c = to_categorical(y_pred, 2, 'pred')
   
    y_pred_c = torch.matmul(y_pred_c, torch.transpose(p, dim0=0, dim1=1)) #loss correction right here
    loss = criterion(y_pred_c, y_actual_c)
    
    return loss 

class LogisticRegression(torch.nn.Module):
    def __init__(self, num_features):
        super(LogisticRegression, self).__init__()
        
        self.fc1 = torch.nn.Linear(num_features, 1, bias=True)
        self.criterion = torch.nn.BCEWithLogitsLoss()
        
    def forward(self, x):
        logits = self.fc1(x)

        return logits
    
    def loss(self, logits, y):
        loss = self.criterion(logits.ravel(), y)
        
        probabilities = torch.sigmoid(logits)
        thresh_results = []
        
        for p in probabilities:
            if p>.5:
                thresh_results.append(1)
            else:
                thresh_results.append(0)
                
        num_correct = 0
        for r,y_ in zip(thresh_results, y):
            if r == y_:
                num_correct += 1
                
        acc = num_correct / len(y)
        
        return loss, acc
    
def train(model, dataset, epsilon, lengths):
    model.train()
    opt = torch.optim.SGD(model.parameters(), lr=.005, weight_decay=0)
    
    criterion1 = torch.nn.BCEWithLogitsLoss()
    criterion2 = torch.nn.BCELoss()
    
    pert_status = np.zeros(len(dataset[0]))
    
    if lengths is not None:
        len_original = lengths[0]
        len_perts = lengths[1]
        total_len = len(dataset[0])
        pert_status = []
        pert_status.extend(np.zeros(len_original))
        pert_status.extend(np.ones(len_perts))
            
    train_data = CreateData(dataset[0], dataset[1], pert_status)
    train_dataloader = DataLoader(train_data, batch_size=1, shuffle=True)

    for itr in range(0, 7):
        itr_loss = 0
        for i, [x,y,p] in enumerate(train_dataloader):
            opt.zero_grad()
            oupt = model(x)
            if p == 0:
                try:
                    loss_val = criterion1(oupt.ravel(), y)
                except ValueError:
                    loss_val = criterion1(oupt, y)
                itr_loss += loss_val
            else:
                loss_val = forward_correct_loss(y.ravel(), oupt, epsilon, criterion2)
                itr_loss += loss_val.item()
            loss_val.backward()
            opt.step() 
        print(f'Epoch {itr+1} Loss: {itr_loss/len(train_dataloader):.3f}')
            
    return model


def calc_influence_single(model, epsilon, train_data, test_data, group_data, device, num_features, criterion):
    start = time.time()
    est_hess = explicit_hess(model, train_data, device, criterion)

    grad_test = grad_z([test_data[0], test_data[1]], model, device, criterion)
    s_test_vec = torch.mm(grad_test[0], est_hess.to(device))

    P = get_p(epsilon)
    
    p_01, p_10 = P[0][1].item(), P[1][0].item()
    
    pi_1 = sum(list(group_data[1]))
    pi_0 = len(group_data[1]) - pi_1
    
    lam_0 = round(p_01 * pi_1)
    lam_1 = round(p_10 * pi_0)

    S_pert = 1 - group_data[1]
    
    y_w_group_pert = pd.concat([group_data[3], S_pert], axis = 0, ignore_index=True)
    y_wo_pert = pd.concat([group_data[3], group_data[1]], axis = 0, ignore_index=True)
    reconstructed_x = pd.concat([group_data[2], group_data[0]], axis = 0, ignore_index=True)
  
    assert len(S_pert) == len(group_data[1])
    grad_z_vec = grad_training([group_data[0],group_data[1]], S_pert, [model], device, [lam_0, lam_1, epsilon], criterion)
  
    influence = torch.dot(s_test_vec.flatten(), grad_z_vec[0].flatten()) * (-(lam_0+lam_1)/len(train_data[0]))
    end = time.time() - start

    return influence.cpu(), end

def explicit_hess(model, train_data, device, criterion):
 
    logits = model(train_data[0])
    loss = criterion(logits.ravel(), train_data[1])
    
    grads = grad(loss, model.parameters(), retain_graph=True, create_graph=True)

    hess_params = torch.zeros(len(model.fc1.weight[0]), len(model.fc1.weight[0]))
    for i in range(len(model.fc1.weight[0])):
        hess_params_ = grad(grads[0][0][i], model.parameters(), retain_graph=True)[0][0]
        for j, hp in enumerate(hess_params_):
            hess_params[i,j] = hp
    
    inv_hess = torch.linalg.inv(hess_params)
    
    return inv_hess

def grad_z(test_data, model, device, criterion):

    model.eval()

    test_data_features = test_data[0]
    test_data_labels = test_data[1]

    logits = model(test_data_features)
    loss = criterion(logits, torch.atleast_2d(test_data_labels).T)
    
    return grad(loss, model.parameters())

def grad_training(train_data, y_perts, parameters, device, epsilon, criterion):
     
    criterion2 = torch.nn.BCELoss()
    
    lam_0, lam_1, ep = epsilon
    lam = lam_0 + lam_1
    len_s = len(y_perts)
    
    train_data_features = torch.FloatTensor(train_data[0].values).to(device)
    train_data_labels = torch.FloatTensor(train_data[1].values).to(device)
    train_pert_data_labels = torch.FloatTensor(y_perts.values).to(device)
    
    model = parameters[0]
    model.eval()

    logits = model(train_data_features)
        
    p = get_p(ep)

    y_actual_c = to_categorical(train_data_labels, 2, 'orig')
    y_pred_c = to_categorical(logits, 2, 'pred')

    orig_loss = criterion(logits, torch.atleast_2d(train_data_labels).T)

    y_pred_c = torch.matmul(y_pred_c, torch.transpose(p, dim0=0, dim1=1)) #loss correction right here
    pert_loss = criterion2(y_pred_c, y_actual_c)

    loss = (lam / len_s)*(pert_loss -  orig_loss)
    print(loss.item())
    to_return = grad(loss, model.parameters())
        
    return to_return

def run_exp(dataset, epsilons, ks, num_rounds):

    device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
    criterion = torch.nn.BCEWithLogitsLoss()
    
    all_orig_loss_e_k = []
    all_est_loss_e_k = []
    all_time = []
    
    for nr in range(num_rounds):
        print(f'\nRound {nr+1}')
        ############
        # Get data #
        ############
        print('\nGetting Data...')
        if dataset == 'adult':
            data = get_adult()
            label = 'income_class'
        elif dataset == 'diabetes':
            data = get_diabetes()
            label = 'readmitted'
        else:
            data = get_law()
            label = 'admit'

        feature_set = set(data.columns) - {label}
        num_features = len(feature_set)
    
        X = data[feature_set]
        y = data[label]

        if dataset == 'diabetes':
            undersample = RandomUnderSampler(random_state=42)
            new_X, new_y = undersample.fit_resample(X, y)
        else:
            new_X = X
            new_y = y

        X_train, X_test, y_train, y_test = train_test_split(new_X, new_y, test_size=0.20, random_state=42)
  
        new_train_df = pd.concat([X_train, y_train], axis=1)
  
        train_sample_num = len(X_train)
    
        x_test_input = torch.FloatTensor(X_test.values).to(device)
        y_test_input = torch.FloatTensor(y_test.values).to(device)

        x_train_input = torch.FloatTensor(X_train.values).to(device)
        y_train_input = torch.FloatTensor(y_train.values).to(device)
   
        ##############################################
        # Train original model and get original loss #
        ##############################################
        print('Training original model...')
        torch_model = LogisticRegression(num_features)
        torch.save(torch_model.state_dict(), 'models/initial_config.pth')
        torch_model.to(device)
        torch_model = train(torch_model, [x_train_input, y_train_input], None, None)
        test_loss_ori, acc_ori = torch_model.loss(torch_model(x_test_input), y_test_input)
        
        e_k_act_losses = []
        e_k_est_losses = []
        influence_time = []
        
        ################################################################
        # Perform influence and retraining for all epsilons a k values #
        ################################################################
        print('\nBegining epsilon and k rounds')
        print('-----------------------------')
        for ep in epsilons:
            print(f'\nEpsilon: {ep}')
            
            k_act_losses = []
            k_est_losses = []
            inf_time = []
            
            for k in ks:
                
                # Influence
                print(f'k: {k}')
                
                selected_group_X, selected_group_y, not_selected_group_X, not_selected_group_y = get_data(new_train_df, feature_set, label, k)

                loss_diff_approx, tot_time = calc_influence_single(torch_model, ep, [x_train_input, y_train_input], [x_test_input, y_test_input], [selected_group_X, selected_group_y, not_selected_group_X, not_selected_group_y], device, num_features, criterion)
                loss_diff_approx = -torch.FloatTensor(loss_diff_approx).cpu().numpy()

                # Retrain
                P = get_p(ep)

                p_01, p_10 = P[0][1].item(), P[1][0].item()

                pi_1 = sum(list(selected_group_y))
                pi_0 = len(selected_group_y) - pi_1

                lam_0 = round(p_01 * pi_1)
                lam_1 = round(p_10 * pi_0)

                S = pd.concat([selected_group_X, selected_group_y], axis=1, ignore_index=False)

                G0 = S[label][S[label].eq(1)].sample(lam_0).index
                G1 = S[label][S[label].eq(0)].sample(lam_1).index

                G = S.loc[G0.union(G1)]
                not_g = S.drop(G0.union(G1))

                G_pert = 1 - selected_group_y

                y_w_group_pert = pd.concat([not_selected_group_y, not_g[label], G_pert], axis = 0, ignore_index=True)
                y_wo_pert = pd.concat([not_selected_group_y, not_g[label], G[label]], axis = 0, ignore_index=True)
                reconstructed_x = pd.concat([not_selected_group_X, not_g[feature_set], G[feature_set]], axis = 0, ignore_index=True)

                model_pert = LogisticRegression(num_features)
                model_pert.load_state_dict(torch.load('models/initial_config.pth'))
                model_pert.to(device)
                model_pert = train(model_pert, [torch.FloatTensor(reconstructed_x.values).to(device), torch.FloatTensor(y_w_group_pert.values).to(device)], ep, [len(not_selected_group_y)+ len(not_g), len(G)])
                test_loss_retrain, acc_retrain = model_pert.loss(model_pert(x_test_input), y_test_input)

                 # get true loss diff
                loss_diff_true = (test_loss_retrain - test_loss_ori).detach().cpu().item()
                
                k_act_losses.append(loss_diff_true)
                k_est_losses.append(loss_diff_approx)
                inf_time.append(tot_time)
            
            e_k_act_losses.append(k_act_losses)
            e_k_est_losses.append(k_est_losses)
            influence_time.append(inf_time)
            
        all_orig_loss_e_k.append(e_k_act_losses)
        all_est_loss_e_k.append(e_k_est_losses) 
        all_time.append(influence_time)
    
    return all_orig_loss_e_k, all_est_loss_e_k, all_time


def main():
    epsilons = [.01, .02, .03, .04, .05, .06, .07, .08, .09, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    k = np.linspace(1, 25, 10)
    rounds = 5
    all_orig_loss_e_k, all_est_loss_e_k, all_time = run_exp('adult', epsilons, k, rounds)
    # [round1[epsilon1[k1,...k10], epsilon2[k1,...k10],...], round2[...]]     

    with open('all_orig_loss_e_k_flc.txt', "wb") as file:   #Pickling
        pickle.dump(all_orig_loss_e_k_flc, file)

    with open('all_est_loss_e_k_fkc.txt', "wb") as file2:   #Pickling
        pickle.dump(all_est_loss_e_k_flc, file2)
        
if __name__ == '__main__':
    main()