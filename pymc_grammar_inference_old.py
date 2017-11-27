from analyze_trans_sim import *
import pymc3 as pm
import theano as T
import theano.tensor as tt

def model(sim_data, prim_data):


    mod = pm.Model()
    with mod:
        probs = pm.Dirichlet("probs", np.ones(N_PRIM), SHAPE=(N_PRIM,1))

        probs = tt.as_tensor(np.ones((N_PAIRS, N_TOP, N_PRIM))/float(N_PRIM))

        ents = tt.pow(probs, prim_data_lst)
        ents = tt.log(ents)
        x1 = tt.sum(ents,axis=2)
        x2 = tt.exp(x1)
        ents = x1 * x2
        ents = tt.sum(ents, axis=1)






        #ll = tt.pow(probs[])




if __name__ == "__main__":
    sim_data =  get_sim_data("out_sim_only.csv")
    prim_data = get_trans_data_by_prim("out_priors.csv")

    sim_data_lst =[]
    prim_data_lst = []

    for k in prim_data.keys():
        prim_data_lst.append(copy.deepcopy(prim_data[k]))

        if k in sim_data:
            sim_data_lst.append(copy.deepcopy(sim_data[k]))

        else:
            sim_data_lst.append(copy.deepcopy(sim_data[(k[1], k[0])]))


    N_PRIM = len(prim_data_lst[0][0])
    N_TOP = len(prim_data_lst[0])
    N_PAIRS = len(prim_data_lst)
    N_PER_PAIR = [len(x) for x in sim_data_lst]

    prim_data_lst = np.array(prim_data_lst)
    sim_data_lst = np.array(sim_data_lst)

    print prim_data_lst.shape
    print sim_data_lst

    #model(sim_data_lst, prim_data_lst)
