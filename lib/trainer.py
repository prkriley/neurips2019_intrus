import sys
from collections import defaultdict
import numpy as np
import tensorflow as tf
import time

import lib.ops
from lib.data import linelen, form_adaptive_batches, form_adaptive_batches_grouped
from lib.oracle import inserts_coo_to_tensor, batch_inserts_to_coo, GenerateReferenceInserts
from lib.inference import SampleReferenceInserts
from lib.util import nested_map, nested_flatten, get_optimized_variables, initialize_uninitialized_variables, make_symbolic_cache

class SampleBasedTrainer:
    def __init__(self, model, sess=None, optimized_variables=None,
                 name=None, verbose=False, is_train=True, initialize=True,
                 sampler_opts=None, optimizer_opts=None, grad_clip=0,
                 **kwargs
                 ):
        """
        An imperative trainer is an object that performs training on batches. Works out-of-graph (in python).
        It is hard-coded to do one thing - sample-based training - but it does that thing well.
        :type model: lib.models.Transformer
        :param sess: tf session to use. tf.get_default_session by default, create new if no default.
        """
        self.model = model
        self.name = name = name or 'trainer_' + model.name
        self.sess = sess = sess or tf.get_default_session() or tf.InteractiveSession()
        self.verbose = verbose

        with tf.name_scope(self.name), tf.variable_scope(self.name) as scope:
            optimized_variables = optimized_variables or get_optimized_variables(model, verbose)
            self.optimized_variables = optimized_variables
            self.step = tf.train.get_or_create_global_step(sess.graph)

            # gradient accumulators (for virtual batch training)
            self.accumulated_grads = [tf.Variable(tf.zeros_like(w), trainable=False, name=w.name[:-2] + '_acc')
                                      for w in optimized_variables]
            self.accumulated_num_batches = tf.Variable(tf.zeros(()), trainable=False, name='num_batches_since_update')
            
            ############
            # step 1: precompute encoder state for all unique input lines
            self.encoder_batch_ph = self.model.make_encoder_batch_ph()
            
            enc = model.encode(self.encoder_batch_ph, is_train)
            self.cached_enc_state, self.compute_enc_state = make_symbolic_cache(enc)
                        
            ############
            # step 2: path_sampler samples a batch of trajectories (sequences of inserts)
            # it also caches encoder state for efficiency
            self.path_sampler = SampleReferenceInserts(model, **(sampler_opts or {}), enc_state=self.cached_enc_state)
            #NOTE(prkriley): path_sampler records self.chosen_insert_logprobs of shape [batch], useful for P/Q
            #NOTE(prkriley): already includes masking, so I think the probs correspond to Q?
            #NOTE(prkriley): NO, masking only restricts sampling but does NOT calculate renormalization, so that value is P
            self.cached_enc_state = nested_map(tf.stop_gradient, self.cached_enc_state)
            self.cached_grad_wrt_enc = nested_map(lambda v: tf.Variable(tf.zeros([]), validate_shape=False,
                                                                        trainable=False,
                                                                        name=v.name[:-2] + '_cached_grad'),
                                                  self.cached_enc_state)

            self.reset_cached_grad_wrt_enc = nested_map(lambda acc, tensor: tf.assign(acc, tf.zeros_like(tensor),
                                                                                      validate_shape=False),
                                                        self.cached_grad_wrt_enc, self.cached_enc_state)
            self.fetch_before_batch = tf.group([self.reset_cached_grad_wrt_enc])
            ############
            # step 3: a trajectory is split into slices (for memory efficiency),
            # for each slice we compute dL/d_w_dec and dL/d_enc_state
            self.slice_ph = {
                'out': tf.placeholder('int32', [None, None]),
                'out_len': tf.placeholder('int32', [None]),
                'out_to_inp_indices': tf.placeholder('int32', [None]),
                'ref_len': tf.placeholder('int32', [None]),
                #'ref_inserts': tf.placeholder('int64', [None, 3]),
                #'chosen_inserts': tf.placeholder('int64', [None, 3]),
                'ref_inserts': tf.placeholder('int64', [None, 4]),
                'chosen_inserts': tf.placeholder('int64', [None, 4]),
                'sample_indices': tf.placeholder('int32', [None]),
                'sample_to_inp_indices': tf.placeholder('int32', [None]),
                'relative_positions': tf.placeholder('int32', [None, None, None])
            }
            loss_on_slice, counters_on_slice = self.get_loss_and_counters(
                self.slice_ph, self.cached_enc_state, is_train=is_train,
                **kwargs
            )

            flat_enc_keys = sorted(self.cached_enc_state.keys())
            flat_enc_cache = list(self.cached_enc_state[k] for k in flat_enc_keys)
            flat_accumulated_grad_wrt_enc = [self.cached_grad_wrt_enc[k] for k in flat_enc_keys]

            loss_grads_on_slice = tf.gradients(loss_on_slice, optimized_variables + flat_enc_cache)
            weight_and_enc_grad_accumulators = self.accumulated_grads + flat_accumulated_grad_wrt_enc
            self.update_grads_on_slice = [
                tf.assign_add(grad_acc, grad)
                for grad_acc, grad in zip(weight_and_enc_grad_accumulators, loss_grads_on_slice)
                if grad is not None
            ]
            # ^-- sess.run-ning this will update gradients w.r.t. decoder weights and encoder state

            # accumulators for metrics
            self.accumulated_counters = nested_map(lambda v: tf.Variable(tf.zeros(v.shape, v.dtype), trainable=False),
                                                   counters_on_slice)
            self.update_counters_on_slice = nested_map(tf.assign_add, self.accumulated_counters, counters_on_slice)
            self.fetch_on_slice = tf.group([self.update_grads_on_slice, self.update_counters_on_slice])

            ############
            # step 4: once we're finished with all slices in one batch, it's time we compute the remaining gradients
            # dL/d_w_enc = dL/d_enc_state * d_enc_state/d_w_enc
            
            encoder_state = model.encode(self.encoder_batch_ph, is_train=is_train)
            flat_encoder_state = [encoder_state[k] for k in flat_enc_keys]
            loss_grads_after_slice = tf.gradients(flat_encoder_state, optimized_variables,
                                                  grad_ys=flat_accumulated_grad_wrt_enc)
            self.update_grads_after_batch = [
                tf.assign_add(grad_acc, grad)
                for grad_acc, grad in zip(self.accumulated_grads, loss_grads_after_slice)
                if grad is not None
            ]

            self.fetch_after_batch = tf.group([
                self.update_grads_after_batch,
                tf.assign_add(self.accumulated_num_batches, 1)
            ])

            ############
            # step 5: after one or several batches, we use the accumulated gradients to perform optimization step,
            # compute metrics for summary and then reset all buffers

            with tf.control_dependencies([tf.assert_positive(self.accumulated_num_batches,
                                                             message='Accumulate gradients over at least one '
                                                                     'full batch before averaging them')]):
                loss_denominator = self.get_denominator(self.accumulated_counters)
                self.grads_avg = [grad_acc / loss_denominator for grad_acc in self.accumulated_grads]

            self.opt = self.get_optimizer(self.step, **(optimizer_opts or {}))

            if grad_clip:
                grads, self.grads_global_norm = tf.clip_by_global_norm(self.grads_avg, grad_clip)
            else:
                grads, self.grads_global_norm = self.grads_avg, tf.global_norm(self.grads_avg)

            self.apply_gradients = tf.group(self.opt.apply_gradients(zip(grads, optimized_variables),
                                                                     global_step=self.step))
            self.reset_gradients = tf.group(
                tf.variables_initializer(self.accumulated_grads + [self.accumulated_num_batches]))

            self.compute_metrics = self.aggregate_metrics_from_counters(self.accumulated_counters)
            self.reset_counters = tf.variables_initializer(list(nested_flatten(self.accumulated_counters)))

            if initialize:
                sess.run([self.reset_gradients, self.reset_counters, tf.assign(self.step, 1)])
                remaining_utility_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope.name)
                initialize_uninitialized_variables(sess=sess, var_list=remaining_utility_variables)


    def train_on_batch(self, batch, slice_max_len=None, optimizer_step=True, reset_counters=None, grouped=False, num_samples=1, example_trunc_len=None):
        """
        Accumulates gradients and counters,
        :param batch: a list of pairs [(inp_line, out_line), ...]
        :param slice_max_len: maximum length of a single slice of hypotheses (in tokens)
        :return: total loss
        """
        DEBUG_TIMING = False
        assert optimizer_step or not reset_counters, "do not reset counters if you don't optimize." \
                                                     "Counters contain statistics used for apply_gradients"
        sess, model = self.sess, self.model

        batch = list(batch)
        reference_lengths = [linelen(ref) for _, ref in batch]

        # step 1 save encoder state to tf variables
        enc_feed = self.model.make_feed_dict(batch)
        sess.run(self.compute_enc_state, {self.encoder_batch_ph[k]: enc_feed[k] for k in self.encoder_batch_ph})

        # step 2: sample insert trajectories, cache encoder state
        batch_trajectories = []
        #NOTE(prkriley): figure out index math
          #Number of unique samples: batch_size*samples; batch might have some skipped though
          #Probably can multiply enc_id by num_samples and add i
        #samples_to_inp_indices = [-1 for _ in range(num_samples*len(batch))]
        timestamp = time.time()
        for i in range(num_samples):
          these_trajectories = list(self.path_sampler.generate_trajectories(batch, sess, truncate_to=example_trunc_len))
          #print('Trajectories:\n{}'.format(these_trajectories))
          for t in these_trajectories:
            t['sample_indices'] = i
            #t['sample_indices'] = i + t['out_to_inp_indices']*num_samples
            #samples_to_inp_indices[t['sample_indices']] = t['out_to_inp_indices']
            #NOTE(prkriley): should we do this by slice or something?

          batch_trajectories += these_trajectories
        if DEBUG_TIMING:
            print("Trajectory sampling time: {}".format(time.time() - timestamp))
        #NOTE(prkriley): batch_trajectories elements will have the logp_any_ref in them somewhere
        #NOTE(prkriley): length of batch_trajectories is less than batch_size*max_len
          
        timestamp = time.time()
        sess.run(self.fetch_before_batch)
        if DEBUG_TIMING:
            print("fetch_before_batch_time: {}".format(time.time() - timestamp))

        # step 3: process hypos with decoder, accumulate gradients at encoder
        # 3.1 split data into slices

        #TODO(prkriley): determine how to figure out which entries are the same hypothesis; perhaps an id field in generate_trajectories?
        #NOTE(prkriley): yes, out_to_inp_indices is just a list with the id for each element
        #TODO(prkriley): make sure that all elements for the same input show up in a single slice

        timestamp = time.time()
        if slice_max_len is None:
            slices = [batch_trajectories]
        else:
            #TODO(prkriley): figure out how to do accumulation properly with slicing; may be impossible!!
            #NOTE(prkriley): unless modify form_adaptive batches to put those from same entry in same slice...
            if grouped:
                slices = form_adaptive_batches_grouped(batch_trajectories,
                                               slice_max_len,
                                               cost_func=lambda row: linelen(row['hypo'])**2, num_samples=num_samples)
            else:
                slices = form_adaptive_batches(batch_trajectories,
                                               slice_max_len,
                                               cost_func=lambda row: linelen(row['hypo'])**2)
        if DEBUG_TIMING:
            print("Slicing time: {}".format(time.time() - timestamp))

        # 3.2 process hypos one slice at a time
        for i,slice in enumerate(slices):

            slice_feed = {key: [row[key] for row in slice] for key in slice[0].keys()}

            slice_feed['out_len'] = [linelen(hypo) for hypo in slice_feed['hypo']]
            slice_feed['out'] = model.out_voc.to_matrix(slice_feed.pop('hypo')) #TODO(prkriley):modify?
            slice_feed['ref_inserts'] = batch_inserts_to_coo(slice_feed['ref_inserts'], model.out_voc)
            slice_feed['chosen_inserts'] = batch_inserts_to_coo(slice_feed['chosen_inserts'], model.out_voc)
            slice_feed['ref_len'] = [reference_lengths[i] for i in slice_feed['out_to_inp_indices']]
            sample_to_inp_indices_dict = {}
            for enc_id, sample_id in zip(slice_feed['out_to_inp_indices'], slice_feed['sample_indices']):
              sample_to_inp_indices_dict[sample_id] = enc_id
            sample_indices, sample_to_inp_indices = zip(*sorted(sample_to_inp_indices_dict.items()))
            #assert list(sample_indices) == list(range(len(sample_indices))), "Sample_indices: {}".format(sample_indices)
            slice_feed['sample_to_inp_indices'] = sample_to_inp_indices
            #TODO(prkriley): figure out what to do given that some matrices are smaller; expand all to PxP, P=max(out_len)+1
            P = max(slice_feed['out_len'])+1
            for i in range(len(slice_feed['relative_positions'])):
                #slice_feed['relative_positions'][i] = slice_feed['relative_positions'][i].resize(P,P)
                size = slice_feed['out_len'][i]+1
                assert slice_feed['relative_positions'][i].shape == (size, size)
                #TODO(prkriley): I think having bogus values at ends of dimensions is currently a problem, need to change attention masks to get rid of bogus
                slice_feed['relative_positions'][i] = np.pad(slice_feed['relative_positions'][i], ((0,P-size),(0,P-size)))
            slice_feed['relative_positions'] = np.stack(slice_feed['relative_positions'], axis=0)
            #slice_feed['relative_positions'] = np.stack([m.resize(P,P) for m in slice_feed['relative_positions']], axis=0)

            #NOTE(prkriley): this accumulates gradients from this slice, which are optimized later; fetch_on_slice has side-effects of doing that
            #NOTE(prkriley): to determine whether [batch] has all timesteps, look into slice_feed; UPDATE: it does
            timestamp = time.time()
            sess.run(self.fetch_on_slice, {self.slice_ph[k]: slice_feed[k] for k in self.slice_ph})
            if DEBUG_TIMING:
                print("Slice {} processing time: {}".format(i, time.time() - timestamp))

        # step 4. compute remaining gradients through encoder
        encoder_feed = self.model.make_feed_dict(batch)

        #TODO(prkriley): look at what these gradients are and whether they need any P/Q business
        timestamp = time.time()
        sess.run(self.fetch_after_batch,
                 {self.encoder_batch_ph[k]: encoder_feed[k] for k in self.encoder_batch_ph})
        if DEBUG_TIMING:
            print("fetch_after_batch time: {}".format(time.time() - timestamp))

        metrics = sess.run(self.compute_metrics)

        if optimizer_step:
            sess.run(self.apply_gradients)
            sess.run(self.reset_gradients)
        if reset_counters is None:
            reset_counters = optimizer_step
        if reset_counters:
            sess.run(self.reset_counters)

        return metrics

    def get_loss_and_counters(self, batch_ph, cached_enc_state, is_train,
                              eos_coeff=None, entropy_reg=0.0, loss_use_logp_any_ref=True, loss_use_PQ=False, loss_scale_PQ=True):
        #NOTE(prkriley): I believe batch_ph should have the logp_any_ref key which should be a list, I THINK for every timestep for each batch entry
        #NOTE(prkriley): OK, not sure whether entries are lists now...
        #NOTE(prkriley): logp is definitely the whole batch, but not sure about time; I suspect one timestep
        #NOTE(prkriley): is it possible that the [batch] dim is actually [batch*time]? No real evidence of this, just brainstorming
        #NOTE(prkriley): confirmed: batch dimension is something less than [batch*time]; there is an id field (out_to_inp_indices) indicating which is which; need to accumulate P/Q accordingly

        # encode with cached enc state
        assert not loss_use_PQ
        enc_batch_size = tf.shape(cached_enc_state['out'])[0]
        with tf.control_dependencies([tf.assert_equal(tf.shape(tensor)[0], enc_batch_size)
                                      for tensor in nested_flatten(cached_enc_state)]):
            #NOTE(prkriley): this is the id logic for reordering, can do something similar for accumulation for P/Q
            enc_reordered = {k: tf.gather(v, batch_ph['out_to_inp_indices'])
                             for k, v in cached_enc_state.items()}

        #TODO(prkriley): determine why compute_action_logprobs is called here AND in SampleReferenceInserts
        #NOTE(prkriley): something about train vs. inference? doesn't make sense
        #NOTE(prkriley): constructor for SampleReferenceInserts calls compute_action_logprobs but doesn't use it, so likely TF-lazy
        logp = self.model.compute_action_logprobs(batch_ph, is_train=is_train, enc=enc_reordered)
        insert_logprobas = logp['insert']  # [batch, T, nout, voc_size] (T = nout)
        finish_logprobas = logp['finish']  # [batch, T]

        # get reference inserts
        is_ref_insert = inserts_coo_to_tensor(batch_ph['ref_inserts'],
                                              tf.shape(batch_ph['out']),
                                              len(self.model.out_voc)) # [batch_size, T, nout, voc_size]
        is_chosen_insert = inserts_coo_to_tensor(batch_ph['chosen_inserts'],
                                                 tf.shape(batch_ph['out']),
                                                 len(self.model.out_voc))

        # compute log-probability of any reference insert
        neg_inf_like_logp = tf.fill(tf.shape(insert_logprobas), -1e9) # [batch, T, nout, voc_size] (T = nout = P-1)
        ref_logp = tf.where(is_ref_insert, insert_logprobas, neg_inf_like_logp)
        chosen_logp = tf.where(is_chosen_insert, insert_logprobas, neg_inf_like_logp)

        #TODO(prkriley): make sure loss_use_logp_any_ref is False
        #NOTE(prkriley): log(P/Q) is tf.reduce_logsumexp(ref_logp, axis=(1,2)) which is shape [batch_size]
        #NOTE(prkriley): still need to accumulate, duplicate, stop-grad, and multiply by logp_ref_inserts
        #NOTE(prkriley): look into tf.unsorted_segment_sum; I think enc_batch_size is true batch size
        #TODO(prkriley): should this be einsum or not?
            #NO: maximize log of SUM of probabilities
        #logp_ref_inserts = tf.reduce_logsumexp(ref_logp if loss_use_logp_any_ref else chosen_logp, axis=(1, 2))
        logp_ref_inserts = tf.reduce_logsumexp(ref_logp if loss_use_logp_any_ref else chosen_logp, axis=(2, 3))
        # ^-- [batch_size, T]

        should_finish = tf.reduce_any(is_ref_insert[:, :, :, self.model.out_voc.eos], axis=-1) # [batch, T]

        if loss_use_PQ:
            log_PQ_ratio = tf.where(should_finish, finish_logprobas, tf.reduce_logsumexp(ref_logp, axis=(1,2))) #[batch_size] = [enc_batch_size*num_samples*timesteps]
            #log_PQ_ratio = tf.Print(log_PQ_ratio, [log_PQ_ratio], "log_PQ_ratio: ")
            #log_PQ_ratio = tf.Print(log_PQ_ratio, [batch_ph['out_to_inp_indices']], "out_to_inp_indices: ")

            #TODO(prkriley): for multi sample, need PQ for each sample separately, so need sample id array
            #TODO(prkriley): what would it take to make this a sorted segment sum?
            log_PQ_ratio_by_sample = tf.segment_sum(log_PQ_ratio, batch_ph['sample_indices']) #timesteps removed, [enc_batch_size*num_samples]
        #TODO(prkriley): I think I have some dims wrong in the two batch_ph index arrays; double-check
        if loss_use_PQ and loss_scale_PQ:
          #sum by unique enc+samp id, normalize by enc id
          #if sample_indices is unique, packed, and sorted, we can do:
              #this is a vector of length num_samples*enc_batch_size, contains log PQ for whole sequence, sorted by encoder id
              #need to logsumexp for enc_id based on sample_to_inp_indices; can probably implement ourselves
          log_PQ_maxes = tf.segment_max(log_PQ_ratio_by_sample, batch_ph['sample_to_inp_indices'])
          expanded_log_PQ_maxes = tf.gather(log_PQ_maxes, batch_ph['sample_to_inp_indices'])
          shifted_PQ_exps = tf.exp(log_PQ_ratio_by_sample - expanded_log_PQ_maxes) #[enc_batch_size*num_samples]
          shifted_PQ_logsumexps = tf.log(tf.segment_sum(shifted_PQ_exps, batch_ph['sample_to_inp_indices'])) #[enc_batch_size]
          PQ_logsumexps_by_sample = tf.gather(shifted_PQ_logsumexps + log_PQ_maxes, batch_ph['sample_to_inp_indices'])
          normalized_log_PQ_ratio_by_sample = log_PQ_ratio_by_sample - PQ_logsumexps_by_sample 
          expanded_log_PQ = tf.gather(normalized_log_PQ_ratio_by_sample, batch_ph['sample_indices'])

          
        elif loss_use_PQ:
          expanded_log_PQ = tf.gather(log_PQ_ratio_by_sample, batch_ph['sample_indices'])
        #expanded_log_PQ = tf.Print(expanded_log_PQ,[expanded_log_PQ], "expanded_log_PQ: ")
        #accumulated_log_PQ_ratio = tf.unsorted_segment_sum(log_PQ_ratio, batch_ph['out_to_inp_indices'], enc_batch_size) #[enc_batch_size]
        #accumulated_log_PQ_ratio = tf.Print(accumulated_log_PQ_ratio, [accumulated_log_PQ_ratio], "accumulated_log_PQ_ratio: ")



        #scale up so max is e20 (NOTE(prkriley): behavior changing)
        #if loss_scale_PQ:
          #scale_val = tf.reduce_max(expanded_log_PQ) - 20
          #expanded_log_PQ -= scale_val
          #expanded_log_PQ = expanded_log_PQ / 10
          #expanded_log_PQ += 10

        #expanded_PQ = tf.stop_gradient(tf.gather(tf.exp(accumulated_log_PQ_ratio), batch_ph['out_to_inp_indices'])) #[batch_size]
        if loss_use_PQ:
            expanded_PQ = tf.stop_gradient(tf.exp(expanded_log_PQ))
        #expanded_PQ = tf.Print(expanded_PQ, [expanded_PQ], "expanded PQ: ")

        #normalize to sum to 1
        #if loss_scale_PQ:
          #TODO(prkriley): bug: reduce_sum here is over wrong things, should be exp of sum (sum of exp?) of accumulated_log_PQ_ratio
          #expanded_PQ = expanded_PQ / tf.reduce_sum(expanded_PQ)
        #  pass


        xent_values = -tf.where(should_finish, finish_logprobas, logp_ref_inserts)
        # ^-- [batch_size]
        if loss_use_PQ:
          xent_values = xent_values * expanded_PQ

        # reweighting
        #NOTE(prkriley): I believe this is always None, though may want to check programmatically; possible it is set in a function call without keyword
        if eos_coeff is None:
            #TODO(prkriley): scale by P/Q values
            #NOTE(prkriley): I think this is one timestep, but we won't know actual values until the end? FIND THE TIMESTEP CONTROL LOOP
            xent_numerator = tf.reduce_sum(xent_values)
            #xent_numerator = tf.Print(xent_numerator, [xent_numerator], "xent_numerator: ")
        else:
            print('WARNING: eos_coeff not None! TODO(prkriley): fix PQ case here')
            samples_per_line = tf.to_float(batch_ph['ref_len'])
            weights = tf.where(should_finish,
                               eos_coeff * samples_per_line,
                               (1.0 - eos_coeff) * samples_per_line / (samples_per_line - 1.0))
            # ^-- [batch_size]
            xent_numerator = tf.reduce_sum(xent_values * weights)

        batch_size = tf.shape(insert_logprobas)[0]
        counters = dict(
            batch_size=tf.to_float(batch_size),
            xent_numerator=xent_numerator,
        )

        # assemble loss (crossentropy with some extra steps)
        loss_numerator = xent_numerator

        #NOTE(prkriley): I believe entropy_reg is always 0 for us
        if entropy_reg != 0.0:
            insert_probas = tf.exp(insert_logprobas)  # [batch_size, nout, voc_size]
            insert_p_logp_sum = tf.reduce_sum(insert_probas * insert_logprobas, axis=2)  # [batch_size, nout]

            mask = lib.ops.infer_mask(batch_ph['out'], self.model.out_voc.eos, dtype=tf.float32)  # [batch_size, nout]
            insert_p_logp_sum = tf.reduce_sum(insert_p_logp_sum * mask, axis=1)  # [batch_size]

            finish_p_logp_sum = finish_logprobas * tf.exp(finish_logprobas)  # [batch_size]

            entropy_values = - finish_p_logp_sum - insert_p_logp_sum  # [batch_size]
            entropy_numerator = tf.reduce_sum(entropy_values)

            loss_numerator -= entropy_reg * entropy_numerator
            counters.update(entropy_numerator=entropy_numerator)

        # metrics
        p_correct_numerator = tf.reduce_sum(tf.exp(-xent_values)) #TODO(prkriley): determine how this is used; with PQ it is no longer semantically accurate
        T = tf.shape(insert_logprobas)[1]
        argmax_flat = tf.argmax(tf.reshape(insert_logprobas, [batch_size, T, -1]), axis=-1)
        #TODO(prkriley): this is rank 1, should be 2
        is_argmax_correct = tf.gather_nd(tf.reshape(is_ref_insert, [batch_size, T, -1]),
                tf.stack([tf.broadcast_to(tf.range(batch_size)[:, None], [batch_size, T]), tf.broadcast_to(tf.range(T)[None, :], [batch_size, T]), tf.to_int32(argmax_flat)], -1))

        is_argmax_correct = tf.where(should_finish, tf.exp(finish_logprobas) >= 0.5, is_argmax_correct)

        acc_numerator = tf.reduce_sum(tf.to_float(is_argmax_correct))
        counters.update(
            loss_numerator=loss_numerator,
            acc_numerator=acc_numerator,
            p_correct_numerator=p_correct_numerator,
        )

        return loss_numerator, counters

    def aggregate_metrics_from_counters(self, counters, numerator_suffix='_numerator'):
        """ Compute any utility metrics given accumulated counters from self.get_loss_and_counters(...)[-1]"""
        results = {
            key[:-len(numerator_suffix)]: counters[key] / self.get_denominator(counters)
            for key in counters if key.endswith(numerator_suffix)
        }
        results['grad_norm'] = self.grads_global_norm
        results['step'] = self.step
        if hasattr(self.opt, '_lr'):
            results['learning_rate'] = tf.identity(self.opt._lr)
        return results

    def get_denominator(self, accumulated_counters):
        """ return total batch size as loss denominator """
        return accumulated_counters['batch_size']

    def get_optimizer(self, step, base_lr=1e-4, warmup_time=None, **kwargs):
        if self.verbose:
            if len(kwargs):
                print("OPTIMIZER OPTS:", kwargs)
            print("base_lr={}, warmup_time={}".format(base_lr, warmup_time))
        step = tf.to_float(step)
        learning_rate = base_lr
        if warmup_time is not None:
            learning_rate *= tf.minimum(
                                tf.to_float(step + 1) ** -0.5 * warmup_time ** 0.5,
                                tf.to_float(step + 1) / warmup_time)

        return tf.contrib.opt.LazyAdamOptimizer(learning_rate, **kwargs)


class FixedOrderTrainer(SampleBasedTrainer):
    def __init__(self, *args, mode='random', sampler_opts=None, **kwargs):
        """
        An imperative trainer is an object that performs training on batches. Works out-of-graph (in python).
        It is hard-coded to do one thing - sample-based training - but it does that thing well.
        :type model: lib.models.Transformer
        :param sess: tf session to use. tf.get_default_session by default, create new if no default.
        """
        super().__init__(*args, **kwargs)
        # Don't pass sampler_opts as we change it later any way

        self.path_sampler = GenerateReferenceInserts(self.model.out_voc, mode=mode, **(sampler_opts or {}))

    def get_loss_and_counters(self, batch_ph, cached_enc_state, is_train, loss_use_logp_chosen=False, eos_coeff=None, **kwargs):

        # encode with cached enc state
        enc_batch_size = tf.shape(cached_enc_state['out'])[0]
        with tf.control_dependencies([tf.assert_equal(tf.shape(tensor)[0], enc_batch_size)
                                      for tensor in nested_flatten(cached_enc_state)]):
            enc_reordered = {k: tf.gather(v, batch_ph['out_to_inp_indices'])
                             for k, v in cached_enc_state.items()}

        logp = self.model.compute_action_logprobs(batch_ph, is_train=is_train, enc=enc_reordered)
        insert_logprobas = logp['insert']  # [batch, T, nout, voc_size]
        finish_logprobas = logp['finish']  # [batch, T]
        #finish_logprobas = tf.Print(finish_logprobas, [finish_logprobas[0]], "finish_logprobas[0]: ", summarize=1000)

        # get reference inserts
        is_ref_insert = inserts_coo_to_tensor(batch_ph['ref_inserts'],
                                              tf.shape(batch_ph['out']),
                                              len(self.model.out_voc)) # [batch, T, nout, voc_size]
        is_chosen_insert = inserts_coo_to_tensor(batch_ph['chosen_inserts'],
                                                 tf.shape(batch_ph['out']),
                                                 len(self.model.out_voc))

        #is_chosen_insert = tf.Print(is_chosen_insert, [is_chosen_insert[0]], "is_chosen_insert[0]: ", summarize=1000)
        mask_correct = is_chosen_insert if loss_use_logp_chosen else is_ref_insert
        #mask_correct = tf.Print(mask_correct, [mask_correct[0]], "mask_correct[0]: ", summarize=1000)

        # assumes that reference inserts for ended hypo are EOS tokens and after-reference are NULL
        should_finish = tf.reduce_any(is_ref_insert[:, :, :, self.model.out_voc.eos], axis=-1) # [batch, T]
        #should_finish = tf.Print(should_finish, [should_finish[0]], "should_finish[0]: ", summarize=1000)

        #TODO(prkriley): fix
        #TODO(prkriley): why is this einsum and not reduce_logsumexp? ANSWER: only one ref so only bt actual values
            #ALSO does the division on the following lines cover it?
            #This is a sum of log probabilities, which is the same as product of probabilities, which is NOT what we really want (though moot because only one ref)
        #proposal:
        #print("WARNING: Using updated calculation of probabilities, with commented division by num correct")
        print("WARNING: Using baseline but possibly wrong einsum calculation for multi-ref probabilities")
        #logp_ref = tf.where(mask_correct, insert_logprobas, tf.fill(tf.shape(insert_logprobas), -1e9))
        #logp_ref = tf.Print(logp_ref, [logp_ref[0]], "logp_ref[0]", summarize=1000)
        #logp_ref = tf.reduce_logsumexp(logp_ref, axis=(2,3))

        logp_ref_old = tf.einsum("btnl,btnl->bt", insert_logprobas, tf.to_float(mask_correct))
        # equivalent to tf.reduce_sum(insert_logprobas * mask_correct, (1, 2)), but without tmp tensor

        #NOTE(prkriley): T dimension IS sanitary because mask_correct is fully good
        #xent_values = logp_ref
        xent_values = logp_ref_old / (tf.reduce_sum(tf.to_float(mask_correct), (-2, -1)) + 1e-5)
        # logp_ref is divided by number of correct labels to properly compute xent

        xent_values = -tf.where(should_finish,
                                finish_logprobas,
                                xent_values)
        #xent_values = tf.Print(xent_values, [xent_values[0]], "xent_values[0]: ", summarize=1000)
        # ^-- [batch_size, T]

        if eos_coeff is None:
            xent_numerator = tf.reduce_sum(xent_values)
        else:
            samples_per_line = tf.to_float(batch_ph['ref_len'])
            weights = tf.where(should_finish,
                               eos_coeff * samples_per_line,
                               (1.0 - eos_coeff) * samples_per_line / (samples_per_line - 1.0))
            # ^-- [batch_size]
            xent_numerator = tf.reduce_sum(xent_values * weights)

        batch_size = tf.shape(insert_logprobas)[0]
        counters = dict(
            batch_size=tf.to_float(batch_size),
            xent_numerator=xent_numerator,
        )

        # assemble loss (crossentropy)
        loss_numerator = xent_numerator

        # metrics
        #p_correct_numerator = tf.reduce_sum(tf.exp(logp_ref))
        p_correct_numerator = tf.reduce_sum(tf.exp(logp_ref_old))
        T = tf.shape(insert_logprobas)[1]
        argmax_flat = tf.argmax(tf.reshape(insert_logprobas, [batch_size, T, -1]), axis=-1)
        #NOTE(prkriley): T dimension not sanitary, suffix is bogus
        
        # [batch_size, T, nout*voc_size]
        is_argmax_correct = tf.gather_nd(tf.reshape(is_ref_insert, [batch_size, T, -1]),
                tf.stack([tf.broadcast_to(tf.range(batch_size)[:,None], [batch_size, T]), tf.broadcast_to(tf.range(T)[None,:], [batch_size, T]), tf.to_int32(argmax_flat)], -1))
        is_argmax_correct = tf.where(should_finish, tf.exp(finish_logprobas) >= 0.5, is_argmax_correct)


        acc_numerator = tf.reduce_sum(tf.to_float(is_argmax_correct))
        counters.update(
            loss_numerator=loss_numerator,
            acc_numerator=acc_numerator,
            p_correct_numerator=p_correct_numerator,
        )

        return loss_numerator, counters
