from analyze_trans_sim import *
import pymc3 as pm
import theano as T
import theano.tensor as tt

def model(sim_data, prim_data):


    mod = pm.Model()
    with mod:

        alpha = pm.Normal('alpha', mu=0, sd=10)
        beta = pm.Normal('beta', mu=0, sd=10)
        sigma = pm.HalfNormal('sigma', sd=1)

        # Expected value of similarity data
        mu = alpha + beta*prior_data_lst_assign

        Y_obs = pm.Normal('Y_obs', mu=mu, sd=sigma, observed=sim_data_lst)

        #ll = tt.pow(probs[])

        trace = pm.sample(MCMC_STEPS,
            tune=BURNIN,target_accept=0.9, thin=MCMC_THIN)

    map_estimate = pm.find_MAP(model=mod)

    print map_estimate

if __name__ == "__main__":
    #globals
    MCMC_STEPS = 1000
    BURNIN=10
    MCMC_THIN = 1


    #similarity data
    sim_data =  get_sim_data("out_sim_only.csv")
    #prior on MDL transformation
    prior_data = get_trans_data("out_prior2.csv")


    #take data from dictionary form into array form
    sim_data_lst =[]
    prior_data_lst = []
    for k in prior_data.keys():
        prior_data_lst.append(copy.deepcopy(prior_data[k]))
        if k in sim_data:
            sim_data_lst.append(copy.deepcopy(sim_data[k]))
        else:
            sim_data_lst.append(copy.deepcopy(sim_data[(k[1], k[0])]))


    #how many conditions
    N_PAIRS = len(prior_data_lst)
    #how many participants in each condition
    N_PER_PAIR = [len(x) for x in sim_data_lst]

    #which prior is associated with which 
    assigns= []
    for i in xrange(len(N_PER_PAIR)):
        a = [i for _ in xrange(N_PER_PAIR[i])]
        assigns.append(a)
    assigns = np.concatenate(assigns)


    prior_data_lst = np.array(prior_data_lst) 
    sim_data_lst = np.array(sim_data_lst)

    mean_pr = np.mean(prior_data_lst)
    std_pr = np.std(prior_data_lst)
    prior_data_lst = (prior_data_lst - mean_pr)/std_pr
    prior_data_lst_assign = prior_data_lst[assigns]

    flattened_sim = np.sum(sim_data_lst)
    mean_sim = np.mean(flattened_sim)
    std_sim = np.std(flattened_sim)
    sim_data_lst = (flattened_sim - mean_sim)/std_sim

    sim_data_lst = flattened_sim
    print len(prior_data_lst_assign), len(sim_data_lst)

    model(sim_data_lst, prior_data_lst_assign)
