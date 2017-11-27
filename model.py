
from LOTlib.Primitives import primitive
from LOTlib.Miscellaneous import Infinity
from LOTlib.Grammar import Grammar
from collections import defaultdict

from LOTlib.Miscellaneous import attrmem
from LOTlib.Hypotheses.LOTHypothesis import LOTHypothesis
from LOTlib.Hypotheses.Likelihoods.BinaryLikelihood import BinaryLikelihood

from LOTlib.Miscellaneous import Infinity, beta, attrmem
from LOTlib.FunctionNode import FunctionNode
from math import log, exp
import editdistance
from helpers import hamming_distance


##############################################################


##################################################################

MAX = 20
INF = 20



@primitive
def repeat(x):
    if len(x) >= MAX or len(x) < 1:
        return x
    else:
        out =""
        while len(out) < MAX:
            out += x

        return out[:MAX]

@primitive
def append(s1, s2, rec=1):
    if len(s1) >= MAX:
        return s1
    else:
        out = (s1 + s2)[:MAX]
        if rec == 0:
            return out
        else:
            return append(out, s2, rec-1)

@primitive
def delete(s1, n, rec=1):
    length  = len(s1)
    if length >= MAX or length <= n:
        return s1
    else:
        out = s1[:n] + s1[n+1:]
        if rec == 0:
            return out
        else:
            return delete(out, n, rec-1)




@primitive
def insert_every(s1, s2, n, offset=0):
    if n == 0:
        return s1
    elif len(s1) == 0:
        return s2
    elif len(s2) == 0:
        return s1

    else:
        ind1 = 0
        out = ""
        while (ind1 < len(s1) and len(out) < MAX):
            if (ind1 +offset) % n == 0:
                out += s2

            out += s1[ind1]
            ind1 += 1
        return out[:MAX]


@primitive
def delete_every(s, n, offset = 0):
    if n >= len(s):
        return s
    new_out = ""
    for i in xrange(len(s)):
        if (i + offset) % n != 0:
            new_out += s[i]
    return new_out


@primitive
def swap(s1, n):
    length  = len(s1)
    if length >= MAX or length <= n:
        return s1
    else:
        if s1[n-1] == "0":
            repl = "1"
        else:
            repl = "0"
        out = s1[:n] + repl + s1[n+1:]
        return out




@primitive
def invert(x):
    inv = ""
    for i in x:
        if i == "0":
            inv += "1"
        else:
            inv += "0"
    return inv

@primitive
def cut(x, lower, upper):
    return x[lower:upper]
##################################################################
grammar = Grammar(start='SEQ')

for i in xrange(0,15):
    grammar.add_rule('INT', str(i), None, 1.0)

#for i in xrange(1,5):
  #  grammar.add_rule('REC', str(i), None, 1.0/i)

grammar.add_rule('SEQ', "'1'", None, 1.0)
grammar.add_rule('SEQ', "'0'", None, 1.0)

#grammar.add_rule('SEQ', "repeat(%s)", ["TERM"], 1.0)
grammar.add_rule('SEQ', "repeat(%s)", ["SEQ"], 1.0)
grammar.add_rule('SEQ', "append(%s, %s)", ["SEQ", "SEQ"], 1.0)
#grammar.add_rule('SEQ', "append(%s, %s)", ["SEQ", "TERM"], 1.0)
#grammar.add_rule('SEQ', "append(%s, %s)", ["TERM", "SEQ"], 1.0)
grammar.add_rule('SEQ', "swap(%s, %s)", ["SEQ", "INT"], 1.0)
grammar.add_rule('SEQ', "delete(%s, %s)", ["SEQ", "INT"], 1.0)
grammar.add_rule('SEQ', 'invert', ['SEQ'], 1.0) 
grammar.add_rule('SEQ', 'cut(%s, %s, %s)', 
    ["SEQ", "INT", "INT"], 1.0) 

grammar.add_rule('SEQ', 'from_seq', None, 10.0) 



"""
grammar.add_rule('SEQ', "append(%s, %s, rec=%s)",
             ["SEQ", "TERM", "REC"], 1.0)
grammar.add_rule('SEQ', "append(%s, %s, rec=%s)",
                 ["TERM", "SEQ", "REC"], 1.0)
grammar.add_rule('SEQ', "delete(%s, %s, rec=%s)",
                     ["SEQ", "INT", "REC"], 1.0)


grammar.add_rule('SEQ', "insert_every(%s, %s, %s, %s)",
                     ["SEQ","SEQ", "INT", "INT"], 1.0)
grammar.add_rule('SEQ', "insert_every(%s, %s, %s)",
                     ["SEQ","SEQ", "INT"], 1.0)
grammar.add_rule('SEQ', "delete_every(%s, %s, %s)",
                     ["SEQ", "INT", "INT"], 1.0)
grammar.add_rule('SEQ', "delete_every(%s, %s)",
                     ["SEQ", "INT"], 1.0)

"""
##################################################################



class MyHypothesis(LOTHypothesis):
    def __init__(self, **kwargs):


        LOTHypothesis.__init__(self, grammar=grammar,
        maxnodes=400, display='lambda from_seq: %s', **kwargs)


    def __call__(self, *args):
        out = LOTHypothesis.__call__(self, *args)
        return out
    

    def compute_single_likelihood(self, datum):
        alpha = datum.alpha
        true_val = datum.output.keys()[0]
        generated = self(*datum.input)

        generated = generated[:len(true_val)]


        dist = (hamming_distance(generated, true_val[:len(generated)]) + 
                     len(true_val) - len(generated))

        return dist * log(alpha) + (len(true_val) - dist) * log(1.-alpha)
        #dist = editdistance.eval(true_val, generated)

      #  return (max_len - dist) * log(1.0 - alpha) + dist  * log(alpha)







if __name__ == "__main__":
    from LOTlib.SampleStream import *
    from LOTlib.Inference.Samplers.MetropolisHastings import MHSampler
    from LOTlib.DataAndObjects import FunctionData
    import copy
    from helpers import *

    #to_seq = "001001001001001"


    #from_seq = "00010001000100010001"
    #print delete_every(to_seq, 3)
    alpha = 0.001

    seqs_trans = {}

    steps = 10000

    c1 = vanilla_conditions(True, False)[0:4]
    c2 = vanilla_conditions( False, True)[:1]

    for to_seq in c1:
        for from_seq in c2:
            data = [FunctionData(alpha=alpha,
                     input=[from_seq], output={to_seq: len(to_seq)})]
            h0 = MyHypothesis()
            best_post = None
            best_h = None
            best_ed = None
            s =0 
            while (best_post == None or 
                    (best_ed != 0)):
                for h in MHSampler(h0, data, steps=steps):
                    likelihood = h.likelihood
                    if best_post == None or h.posterior_score >= best_post:
                        best_post = h.posterior_score
                        best_h = copy.deepcopy(h)
                        true_val = to_seq
                        generated = best_h(from_seq)[:len(true_val)]

                        best_ed = (hamming_distance(generated, true_val[:len(generated)]) + 
                                     len(true_val) - len(generated))
                       # best_ed = editdistance.eval(best_h(from_seq), to_seq)
                    if s % 1000 == 0:
                        #print s, to_seq, from_seq
                        #print ("Rand: ", h, h(from_seq), 
                           #     editdistance.eval(h(from_seq), to_seq),
                              #      exp(h.posterior_score))

                        print_star(s,"Best: ", best_h, best_h(from_seq), 
                                    best_ed, to_seq,
                                    exp(best_h.posterior_score))
                        print    


                    s += 1

            seqs_trans[(to_seq, from_seq)] = (copy.deepcopy(best_h), best_h.value.count_subnodes())


    for s in seqs_trans:
        print s, seqs_trans[s][0], seqs_trans[s][1]