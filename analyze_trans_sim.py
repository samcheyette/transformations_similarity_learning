import numpy as np
import scipy.stats as st
import copy

def get_sim_data(file):
	"""
	gets participant data from the
	cleaned up MTURK experiment file
	"""
	f = open(file,"r")
	l = f.readline()
	l = f.readline()

	dct = {}
	while l != "":
		r = l.split(",")
		from_s = r[1]
		to_s = r[2]
		sim = int(r[3])

		if (from_s,to_s) not in dct:
			dct[(from_s,to_s)] = []

		dct[(from_s,to_s)].append(sim)
		l = f.readline()

	return dct


def get_trans_data(file):
	"""
	gets model data on prior/# of transformations 
	from the MDL/lowest prior hypothesis...
	using the LOT model output
	"""
	f = open(file,"r")
	l = f.readline()
	l = f.readline()

	dct = {}
	while l != "":
		r = l.split(",")
		from_s = r[0]
		to_s = r[1]
		prior = float(r[3])

		if (from_s, to_s ) not in dct:
			dct[(from_s,to_s)] = prior
		if prior > dct[(from_s, to_s)]:
			dct[(from_s,to_s)] = prior


		l = f.readline()
	return dct


def get_trans_data_by_prim(file):
	"""
	get data on priors/n_transformations, with
	the number of primitives in each transformation(hypothesis)
	included
	"""

	f = open(file,"r")
	l = f.readline()
	l = f.readline()

	dct = {}
	while l != "":
		r = l.split(",")
		from_s = r[0]
		to_s = r[1]
		h = r[2]
		type_s = r[5]
		name_s  = r[6]
		n_inh = float(r[7].replace("\n",""))
		if (from_s,to_s,h) not in dct:
			dct[(from_s,to_s,h)] = []
		dct[(from_s,to_s,h)].append(n_inh)

		l = f.readline()
	new_dct = {}
	for k in dct:
		new_k = (k[0], k[1])
		if new_k not in new_dct:
			new_dct[new_k] = []
		new_dct[new_k].append(copy.deepcopy(dct[k]))

	return new_dct


def n_prims_of_type(file):
	f = open(file,"r")
	l = f.readline()
	l = f.readline()

	#dct = {}
	lst_each = []
	lst_type = []
	last = ""
	while l != "":
		r = l.split(",")

		prim_type = r[5]
		prim_val = r[6]

		if (prim_type, prim_val) not in lst_each:
			lst_each.append((prim_type,prim_val))

			if prim_type != last:
				lst_type.append(0)
			lst_type = (lst_type[:len(lst_type)-1] + 
						[lst_type[len(lst_type)-1] + 1])

			last = prim_type

		l = f.readline()
	return lst_each, lst_type


def get_transfer_data(file):
    f = open(file, "r")
    l = f.readline()
    l = f.readline()

    tmp_1 = []
    tmp_seqs = []
    data = {}

    while l != "":
        l = l.replace(" ","")
        r = l.split(",")
        which = int(r[5].replace("\n",""))
        time = int(r[2])
        seq = r[1]
        corr = int(r[4])
        l = f.readline()

        if time == 0:
            tmp_seqs.append(seq)

        if which == 1:
            tmp_1.append(corr)
            if time == 14:
                key = tuple(tmp_seqs)
                if key not in data:
                    data[key] = []
                data[tuple(tmp_seqs)].append(copy.deepcopy(tmp_1))

                tmp_1 = []
                tmp_seqs = []


    return data

if __name__ == "__main__":
	prim_type = n_prims_of_type("out_priors2.csv")

	sim_data =  get_sim_data("out_sim_only.csv")
	trans_data = get_trans_data("out.csv")

	trans_data_full = get_trans_data_by_prim("out.csv")

	for t in trans_data_full:
		print t, trans_data_full[t]

	for s in trans_data:
		print s, trans_data[s]


	lst_sim = []
	lst_trans = []
	for s in sim_data:
		for i in xrange(len(sim_data[s])):
			lst_sim.append(sim_data[s][i])
			if s in trans_data:
				lst_trans.append(trans_data[s])
			else:
				lst_trans.append(trans_data[(s[1], s[0])])
		#print s, sum(sim_data[s])/float(len(sim_data[s]))

	print st.pearsonr(lst_sim,lst_trans)

