
import pandas as pd
import copy
import scipy.stats as st
import numpy as np
from collections import defaultdict
import random
from analyze_trans_sim import get_trans_data_by_prim, n_prims_of_type

def get_prior_data(file, keys):
    df =pd.read_csv(file, usecols=keys)
    #
    return df



def get_sim_data(file, keys, keys_norm):
    df = pd.read_csv(file)
    df = df.rename(columns=lambda x: x.strip())
    df = df[keys]
    df.columns = keys_norm
    return df

def get_entropy(d):
    #takes dictionary of log-priors

    ents = {}
    for k in d:
        priors = np.array(d[k])
        pr_exp = np.exp(priors)
        norm = np.sum(pr_exp)
        pr_exp = pr_exp/norm

        norm_prior = priors * pr_exp
        norm_prior = norm_prior.sum()

        ents[k] = norm_prior

    return ents

def analyze_sim_ent(ent, sim):

    ents_lst = []
    sim_lst = []
    for k in ent:
        assert(k in sim)
        ents = np.repeat(ent[k], len(sim[k]))
        ents_lst.append(ents)
        sim_lst.append(sim[k])

    ents_lst = np.concatenate(ents_lst)
    sim_lst = np.concatenate(sim_lst)

    return st.pearsonr(ents_lst, sim_lst)


def main(prior_data, sim_data, gram_data,n_type):


    #gram_probs = np.array([item for sublist in gram_probs
                   #      for item in sublist])


    #random initialization
    gram_probs = []
    max_pears_r = 1e-5
    for n in n_type:
        l = np.random.rand(n)
        l = l/np.sum(l)
        gram_probs.append(l)

    probs_max = copy.deepcopy(gram_probs)

    for _ in xrange(NSAMP):

        dct_gram = defaultdict(list)
        dct_probs = defaultdict(list)

        new_gp = []
        for g in probs_max:
            brk = False
            new_p = np.random.dirichlet(g) + 0.01


            new_gp.append(new_p)
        gram_probs = copy.deepcopy(new_gp)


        for k in gram_data:
            tmp_probs = []

            for l in xrange(len(gram_data[k])):

                last = 0
                tmp = []
                tmp_prob = 0.
                for i in xrange(len(list(n_type))):
                    t = n_type[i]
                    lst = gram_data[k][l]
                    lst_part = np.array(lst[last:last + t]) 
                    last = last + t
                    gram_prob = np.array(gram_probs[i])
                    nonzero = np.where(lst_part > 0.)[0]


                    prob = np.log(gram_prob[nonzero]) * lst_part[nonzero]

                    tmp.append(prob)
                    tmp_prob += np.sum(prob)

                    #tmp.append(np.array(lst_part)/float(t))
                tmp_probs.append(tmp_prob)
                #tmp = [item for sublist in tmp for item in sublist]
                #print tmp
                #sprint list(tmp), sum(tmp)

            dct_probs[k] = copy.deepcopy(tmp_probs)

        """

        prior_dct = {}
        for l in prior_data:
            prior_dct[l] = np.array(prior_data[l].tolist())
        
        """

        sim_dct = {}
        for l in sim_data:
            sim_dct[l] = np.array(sim_data[l].tolist())

        zip_key = zip(sim_dct['from'],sim_dct['to'])
        sim = defaultdict(list)

        for k in xrange(len(zip_key)):
            sim[zip_key[k]].append(sim_dct['similarity'][k])
            sim[(zip_key[k][1], zip_key[k][0])].append(sim_dct['similarity'][k])

        """
        zip_key = zip(prior_dct['from'],prior_dct['to'])
        prior = defaultdict(list)
        for k in xrange(len(zip_key)):
            prior[zip_key[k]].append(prior_dct['prior'][k])
        """

       # prior_ent = get_entropy(prior)
        #print prior_ent
        entr = get_entropy(dct_probs)

        pears_r = analyze_sim_ent(entr, sim)[0]

        ratio = (pears_r/max_pears_r)**5
        if random.random() < 0.05 * ratio:
            max_pears_r =pears_r
            probs_max = copy.deepcopy(gram_probs)

        print max_pears_r

if __name__ == "__main__":
    NSAMP = 5000

    keys_prior = ['from','to','prior']
    keys_sim = ['training','transfer', 'similarity']
    keys_norm_sim = ['from', 'to', 'similarity']
    file_prior = "out_prior2.csv"
    file_sim = "out_sim_only.csv"
    prior_data = get_prior_data(file_prior, keys_prior)
    sim_data = get_sim_data(file_sim, keys_sim, keys_norm_sim)

    #how many primitives of each type
    each_prim, prim_type = n_prims_of_type(file_prior)
    grammar_data = get_trans_data_by_prim(file_prior)


    main(prior_data,sim_data, grammar_data, prim_type)

    

    #prior_stats = get_prior_stats(prior_data)