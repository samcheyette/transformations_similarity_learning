from analyze_trans_sim import *
import pymc3 as pm
import theano as T
import theano.tensor as tt
import pandas as pd
from helpers import *
import os

def model(sim_data, prior_data, keys=[], out_fold='test'):


    #x = tt.as_tensor(np.ones(4))
    #y = tt.as_tensor(np.ones(3))
    #z = tt.concatenate([x,y],axis=0)
    #print z.eval()

    for file in os.listdir(out_fold):
        name = "%s/%s" % (out_fold,file)
        os.remove(name)
        #print file

    mod = pm.Model()
    with mod:

        #probailities of each primitive

        ps = []
        #weights = pm.Dirichlet("weights", np.ones(len(prim_type)))
        for i in xrange(len(prim_type)):
            #weight = weights[i]
            #weight = 1.
            #prob = np.ones(prim_type[i])/float(prim_type[i])


            name = "p_%s" % i
            name_w = name + "_w"
            weight = pm.Exponential(name_w, 1.0) * np.ones(prim_type[i])
            prob = pm.Dirichlet(name, np.ones(prim_type[i]))

            ps.append(weight * prob)

        probs =  tt.concatenate(ps, axis=0)
        
        #probs = np.ones(N_PRIM)/float(N_PRIM)


        #copy the probability vector a number of times
        #so that it becomes a tensor
        #ith that the probabilities of each primtive
        #for each hypothesis for each sequence pair
        probs = tt.tile(probs, (N_TOP, 1))
        probs = tt.tile(probs, (N_PAIRS, 1,1))     

        #and now convert the probabilities assigned 
        #to each hypothesis for a given sequence pair
        #into entropy

        ents = tt.pow(probs, prior_data)
        ents = tt.log(ents)

        """
        x1 = tt.sum(ents,axis=2)
        x2 = tt.exp(x1)
        norms = tt.sum(x2)
        x2 = x2/norms
        ents = -1.0 * x1 * x2
        ents = tt.sum(ents, axis=1)
        """

        ents = tt.sum(ents, axis=2)
        ents = tt.max(ents, axis=1)
        #ents = tt.exp(ents)

        ents = pm.Deterministic('ents', ents)

        mean_pr = tt.mean(ents)
        #ents = ents - mean_pr
        std_pr = tt.std(ents)
        ents = (ents - mean_pr)/std_pr
        ents = ents[assigns]


        #intercept
        #alpha = pm.Uniform('alpha', 0,1) * 5.
        alpha = pm.Normal('alpha', mu=3, sd=1)
        #slope
        beta = pm.Normal('beta', mu=0.0, sd=10)
        #standard deviation
        sigma = pm.HalfNormal('sigma', sd=1)
        #expected value of similarity 
        mu = alpha + beta*ents

        #compare fit to observed similarity data
        Y_obs = pm.Normal('Y_obs', mu=mu, sd=sigma, 
            observed=sim_data_lst)



        db = pm.backends.Text(out_fold)
        trace = pm.sample(MCMC_STEPS,
            tune=BURNIN,
            thin=MCMC_THIN,trace=db)

    map_estimate = pm.find_MAP(model=mod)
    print map_estimate
    c = 0

    if len(keys) > 0:
        for z in map_estimate['ents']:
            print keys[c],z
            c += 1

    return trace




def analyze_trans_learn(gram_probs, prior_data, data, assigns):


    gram_probs = np.tile(gram_probs, (N_TOP, 1))
    gram_probs = np.tile(gram_probs, (N_PAIRS, 1,1))   



    ents = np.power(gram_probs, prior_data)

    ents = np.log(ents)
    ents = np.sum(ents, axis=2)
    ents = np.max(ents, axis=1)

    mean_pr = np.mean(ents)
    std_pr = np.std(ents)
    ents = (ents - mean_pr)/std_pr
    #for ent in ents:
       # print ent
    ents = ents[assigns]

    time = np.array([(i % 15 - 7.5) for i in xrange(len(assigns))])

    df = pd.DataFrame({'entropy':ents,
         'seqs': assigns,
         'time': time,
         'resps': trans_data_lst})

    with pm.Model() as logistic_model:
        pm.glm.GLM.from_formula('resps ~ time + entropy',
                 df, family=pm.glm.families.Binomial())
        trace_logistic_model = pm.sample(100, tune=10)


    map_estimate = pm.find_MAP(model=logistic_model)

    print map_estimate

if __name__ == "__main__":
    #globals
    MCMC_STEPS = 10
    BURNIN=5
    MCMC_THIN = 1
    OUT_FILE = "grammar_probs.csv"

    #transfer data from experiment
    trans_data = get_transfer_data("out_trans_only.csv")
    #similarity data from experiment
    sim_data =  get_sim_data("out_sim_only.csv")
    #priors on transformations (and each primitive in hypothesis)
    #from LOT model
    prior_data = get_trans_data_by_prim("out_prior2.csv")

    #how many primitives of each type
    each_prim, prim_type = n_prims_of_type("out_prior2.csv")

    #turn both into lists (from dictionaries)
    sim_data_lst =[]
    trans_data_lst = []
    trans_data_ns = []
    prior_data_lst = []

    c = 0
    keys_2_list=prior_data.keys()
    for k in prior_data.keys():
        prior_data_lst.append(copy.deepcopy(prior_data[k]))

        for participant in xrange(len(trans_data[k])):
            for time in xrange(len(trans_data[k][participant])):
                resp = trans_data[k][participant][time]
                trans_data_lst.append(resp)
                trans_data_ns.append(c)


        #trans_data_lst.append(np.sum(np.array(trans_data[k]),
                                    #    axis=0))
        #trans_data_ns.append(len(trans_data[k]))

        if k in sim_data:
            sim_data_lst.append(copy.deepcopy(sim_data[k]))
        else:
            sim_data_lst.append(copy.deepcopy(sim_data[(k[1], 
                                            k[0])]))

        c += 1

    #print trans_data_lst
    #print trans_data_ns

    #gram_probs = [np.ones(x)/x for x in prim_type]
    #gram_probs = np.concatenate(gram_probs, axis=0)

    #number of sequence pairs
    N_PAIRS = len(prior_data_lst)
    #number of participants responding per sequence pair
    N_PER_PAIR = [len(x) for x in sim_data_lst]
    #number of top transformation
    #hypotheses recorded per sequence pair
    N_TOP = len(prior_data_lst[0])
    #number of primitives in model
    N_PRIM = len(prior_data_lst[0][0])


    print_star([("N_PAIRS", N_PAIRS),
                 ("N_PER_PAIR",N_PER_PAIR),
                 ("N_TOP", N_TOP),
                 ("N_PRIM", N_PRIM)])


    assigns= []
    for i in xrange(len(N_PER_PAIR)):
        a = [i for _ in xrange(N_PER_PAIR[i])]
        assigns.append(a)

    #this is used to flatten the array of
    #hypotheses, mapping it to each sequence pair 
    #for each participant responding to that 
    #sequence pair
    assigns = np.concatenate(assigns)


    prior_data_lst = np.array(prior_data_lst) 
    sim_data_lst = np.array(sim_data_lst)


    prior_data_lst_assign = prior_data_lst[assigns]

    #flatten similarity data
    sim_data_lst = np.sum(sim_data_lst)

    care_about = {"INT":"p_0", "SEQ":"p_1"}
    order = {}
    for x in each_prim:
        if care_about[x[0]] not in order:
            order[care_about[x[0]]] = []
        order[care_about[x[0]]].append(x)

    fit_sim = model(sim_data_lst, prior_data_lst, keys=keys_2_list)


    #order = ["INT", "SEQ"]
    gram_probs = output_save_gp(fit_sim, order, OUT_FILE)
    gram_probs = np.concatenate([gram_probs['INT'], gram_probs['SEQ']])

    fit_trans = analyze_trans_learn(gram_probs,prior_data_lst,
            trans_data_lst, trans_data_ns)


    output_betas(fit_sim, care_about=["alpha", "beta", "sigma"])