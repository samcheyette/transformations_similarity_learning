
def print_star(*args):
    if len(args) > 0:
        for arg in args:
            print arg
        print "*" * 100
        print


def vanilla_conditions(appstim=True, append=True):
	stims = []
	endings = []
	if appstim:
		stims = ["aabaabaabaabaab", "baabaaabaaaabaa", "aaaaaabbbbbbbbb", "bbaaabbaaabbaaa", "bbbbbababababab",
		             "ababbababbababb"]
		stims = [fromAB(s) for s in stims]
	if append:
		endings = ["aaabaaabaaabaaa", "abaabaaabaaaaba", "bbbbaaaaaaaaaaa", 
						"bbaaaabbaaaabba", "aaaabababababab",
		           "babaababaababaa"]
		endings = [fromAB(s) for s in endings]

	all_use = []
	for k in stims:
		all_use.append(k)
	for k in endings:
		all_use.append(k)

	return all_use


def fromAB(s):
    #inverse of toAB
    ret = ""
    for j in s:
        if j == "a":
            l = "0"
        else:
            l = "1"
        ret += l
    return ret


def toAB(s):
    #inverse of toAB
    ret = ""
    for j in s:
        if j == "0":
            l = "a"
        else:
            l = "b"
        ret += l
    return ret
    
def hamming_distance(a, b):
    assert(len(a) == len(b))
    hd = 0
    for i in xrange(len(a)):
        if a[i] != b[i]:
            hd += 1

    return hd


def output_save_gp(fit_gram_mcmc, order, file):

    means = {}
    for o in order.keys():
        k = order[o][0][0]

        means[k] = [0. for _ in xrange(len(order[o]))]

    #means = [0. for _ in xrange(sum(prim_type))]
    out = "sample,type,name,val\n"
    for i in xrange(len(fit_gram_mcmc)):
        samp = fit_gram_mcmc[i]
        for k in order.keys():
            assert(k in samp)
            for z in xrange(len(samp[k])):
                name_p = order[k][z][1]
                type_p = order[k][z][0]

                val = samp[k][z]
                means[type_p][z] += val/float(len(fit_gram_mcmc))

                out += "%d,%s,%s,%f\n" % (i, type_p, name_p, val)

    o = open(file, "w+")
    o.write(out)
    o.close()

    return means