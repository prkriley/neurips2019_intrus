"""
Utility functions to compute optimal actions for a given source hypo
"""
import random
import tensorflow as tf
import numpy as np
import sys


def get_optimal_inserts(cand, ref):
    """
    :param cand: considers inserting tokens into this list
    :param ref: inserts tokens from this list to keep being a sub-sequence of it
    :returns: list of sets, for each position i in [0, len(cand) + 1),
        a list of elems that can be inserted into cand before position i to get larger substring of ref
    cand must be a sub-sequence of ref
    """
    # core idea (by dimdi-y): for every position i in [0, len(cand) + 1],
    # find shortest prefix in ref s.t. cand[:i] is subsequence of prefix
    # find shortest suffix in ref s.t. cand[i:] is subsequence of suffix
    # one can insert into cand at i any token that is between suffix and prefix

    starts = [0]
    ref_iter = iter(enumerate(ref))
    for cand_item in cand:
        for ref_pos, ref_item in ref_iter:
            if ref_item == cand_item:
                starts.append(ref_pos + 1)
                break
        else:
            print('Bogus cand / ref: {} / {}'.format(cand,ref))
            raise ValueError("cand must be a sub-sequence of ref")

    ends = [len(ref)]
    reverse_ref_iter = iter(reversed(list(enumerate(ref))))
    for cand_item in reversed(cand):
        for ref_pos, ref_item in reverse_ref_iter:
            if ref_item == cand_item:
                ends.append(ref_pos)
                break
        else:
            print('Bogus cand / ref: {} / {}'.format(cand,ref))
            raise ValueError("cand must be a sub-sequence of ref")
    ends = ends[::-1]

    inserts = []
    for i, j in zip(starts, ends):
        inserts.append(set(ref[i: j]))
    return inserts


def is_subseq(x, y):
    """ checks if sequence x is subsequence of sequence y """
    it = iter(y)
    return all(c in it for c in x)


def get_optimal_inserts_slow(cand, ref):
    """ Slower function used to test get_optimal_inserts """
    assert is_subseq(cand, ref)
    inserts = []
    for i in range(len(cand) + 1):
        inserts_i = set()
        for token in set(ref):
            new_cand = list(cand)
            new_cand.insert(i, token)
            if is_subseq(new_cand, ref):
                inserts_i.add(token)
        inserts.append(inserts_i)
    return inserts


def generate_insert_trajectory(ref, hypo=(), choice=lambda hypo, ref, inserts: random.choice(inserts)):
    """
    Samples a sequence of insert from hypo (default = empty sequence) to reference
    :param ref: a sequence of reference tokens (list/tuple)
    :param choice: f(hypo, list_of_inserts) -> chosen insert
        where list_of_inserts is a sequence of pairs (i, token) for hypo.insert(i, token)
    :returns: an iterator over triples (hypo, optimal_inserts, chosen_insert)
    """
    hypo = list(hypo)
    DEBUG_it = -1
    while True:
        DEBUG_it += 1
        try:
          inserts = get_optimal_inserts(hypo, ref)
        except ValueError as e:
          print('Dying in iteration {} of generate_insert_trajectory with ref {}'.format(DEBUG_it, ref), file=sys.stderr)
          raise e
        flat_inserts = [(i, token) for i, tokens in enumerate(inserts) for token in tokens]
        if not len(flat_inserts):
            yield hypo, [set() for _ in range(len(hypo) + 1)], None
            return
        chosen_insert = choice(hypo, ref, flat_inserts)
        yield list(hypo), inserts, chosen_insert
        hypo.insert(*chosen_insert)


def batch_inserts_to_coo(batch_inserts, voc, dtype=np.int64):
    """
    Convert inserts from [sets of tokens] to a coo format
    :param inserts: list of [inserts as produced by get_get_optimal_inserts]
    :returns: matrix [num_inserts, 3] where each row is
        [hypo_index_in_batch, insert_pos, insert_token_ix]
    :rtype: np.ndarray
    """
    return np.asarray([
        (batch_i, timestep_i, pos_i, token_i)
        for batch_i, timesteps_i in enumerate(batch_inserts)
        for timestep_i, inserts_i in enumerate(timesteps_i)
        for pos_i, insert_tokens in enumerate(inserts_i)
        for token_i in sorted(voc.ids(tok) for tok in insert_tokens)
    ], dtype=dtype)


def inserts_coo_to_tensor(inserts_coo, hypo_shape, voc_size, dtype=tf.bool, sparse=False):
    """
    Converts inserts from [[batch_i, insert_pos, insert_token]] format into a 3d tensor
    :param inserts_coo: tf tensor int32[num_inserts, 3]
    :param hypo_shape: shape of token ix matrix to insert into. shape:[batch_size, max_len]
    :param voc_size: number of tokens in insert (out) vocabulary
    :returns: tf tensor of shape [batch_size, max_len, voc_size]
    """
    #ref_shape = tf.stack([hypo_shape[0], hypo_shape[1], voc_size])
    #NOTE(prkriley): need to add timestep dimension, which is same as ref_length which is hypo_shape[1] (maybe +/- 1, verify)
    ref_shape = tf.stack([hypo_shape[0], hypo_shape[1], hypo_shape[1], voc_size])
    inserts_tensor = tf.SparseTensor(
        inserts_coo,
        tf.ones(tf.shape(inserts_coo)[:1], dtype=dtype),
        tf.to_int64(ref_shape))
    if not sparse:
        inserts_tensor = tf.sparse.to_dense(inserts_tensor,
                                            default_value=tf.cast(0, dtype))
    return inserts_tensor


def _map_chosen_to_prod(chosen, prod_order):
    new_prod_order = prod_order.copy()
    if chosen is None:
        i = prod_order[-1]
        new_prod_order.insert(len(prod_order), len(prod_order))
    else:
        i = prod_order[chosen[0]]
        new_prod_order.insert(chosen[0]+1, len(prod_order))
    return i, new_prod_order

def extend_relative_positions_matrix(R, prod_order, chosen):
    #i is production index AFTER which to insert
    i, new_prod_order = _map_chosen_to_prod(chosen,prod_order)
    v = R[i].copy()
    v[v == 1] = 0
    return np.concatenate([np.concatenate((R,v[None,:]),axis=0), np.concatenate([2-v,[1]], axis=0)[:,None]],axis=1), new_prod_order

def relative_positions_matrix_and_prod_order():
    return np.array([[1]]), [0]


class GenerateReferenceInserts:
    def __init__(self, voc, mode='random', samples_per_line=None, **params):
        """
        Uniformly samples trajectory
        :param voc: Vocabulary
        :param params: optional parameters
        """
        self.voc = voc
        self.mode = mode
        self.samples_per_line = samples_per_line
        self.steps_after_eos = 0

    def choose_insert(self, hypo, ref, inserts):
        """
        selects next insertion out of all possible ones
        :param hypo: a list of tokens for current subsequence
        :param ref: a list of tokens for target sequence
        :param inserts: pairs (position, token)
        """
        assert len(inserts) > 0
        if self.mode == 'random':
            return random.choice(inserts)
        elif self.mode == 'l2r':
            chosen_insert = (len(hypo), ref[len(hypo)])
            assert chosen_insert in inserts
            return chosen_insert
        else:
            raise NotImplementedError("Unsupported mode: " + self.mode)


    def generate_trajectories(self, batch, *args, truncate_to=None, **kwargs):
        """
        Samples trajectories that start at empty hypothesis and end on reference lines
        :param batch: a sequence of pairs[(inp_line, ref_out_line)]
        :return: a sequence of dicts {inp_line, hypo, out_to_inp_index, ref_inserts, chosen_inserts, ...}
        """
        inp_lines, ref_lines = zip(*batch)
        out_voc = self.voc
        hypos_ref_tok = [out_voc.words(out_voc.ids(ref_line.split())) for ref_line in ref_lines]
        if truncate_to:
            hypos_ref_tok = [hrf[:truncate_to] for hrf in hypos_ref_tok]

        for i, (inp_line, ref_tok) in enumerate(zip(inp_lines, hypos_ref_tok)):
            trajectory = list(generate_insert_trajectory(ref_tok, choice=self.choose_insert))
            #TODO(prkriley): right here we can collect these timesteps into 1
            if self.samples_per_line is not None:
                raise NotImplementedError
                trajectory = random.sample(trajectory, k=min(len(trajectory), self.samples_per_line))
            #NOTE: need to also put EOS before generated tokens
            #TODO(prkriley): reorder hypo to be actual generation order AND calculate matrix R
                #each chosen is (pos, token)
                #hypo is just those tokens, in order
                #have extend_R(R,i) where i is production index of thing to insert before
                #need to convert the pos's from chosen (which are absolutes) into that
                #hypo is BEFORE doing the insert, so the index is the thing we want
                #NOTE that chosen_index works as an insert-before index if there is implicit EOS at first
                #need to keep track of surface sequence with prod-idx (leave off implicit 0 at beginning)
                    #[1]; c=0
                    #[2 1]; c=1
                    #[2 3 1]
            yield_dict = dict(inp_line=inp_line, out_to_inp_indices=i, ref_inserts=[], chosen_inserts=[])
            R, prod_order = relative_positions_matrix_and_prod_order()
            prod_tokens = []
            for hypo, inserts, chosen in trajectory:
                R, prod_order = extend_relative_positions_matrix(R, prod_order, chosen)
                if chosen is None:  # nothing to insert
                    inserts = [{out_voc.EOS} for _ in inserts]
                    chosen = (random.randint(0, len(inserts)), out_voc.EOS)
                else:
                    prod_tokens.append(chosen[1])
                #TODO(prkriley): determine whether we need to do something special when chosen is none for prod_order
                chosen = [{chosen[1]} if i == chosen[0] else set() for i in range(len(inserts))]
                yield_dict['ref_inserts'].append(inserts)
                yield_dict['chosen_inserts'].append(chosen)

                #TODO(prkriley): hypo could probably be repurposed as chosen_inserts, just with tokens in production order; ref_inserts will have to be 2D (list of list?), just enumeration of sublists existing now
                  #current fields: step, inp_line, hypo, ref_inserts, chosen_inserts, out_to_inp_indices
                  #remove: step, hypo
                  #modify: ref_inserts, chosen_inserts
                    #ref_inserts: concat into 2D
                    #chosen_inserts: concat into 2D (will be post-processed later)
                #yield dict(step=len(hypo), inp_line=inp_line, hypo=' '.join(hypo), ref_inserts=inserts,
                #           chosen_inserts=chosen, out_to_inp_indices=i)
                # yeah, select randomly again as we don't care if it's the same
            yield_dict['hypo'] = ' '.join(prod_tokens)
            yield_dict['relative_positions'] = R
            yield yield_dict
