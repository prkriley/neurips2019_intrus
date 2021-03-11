"""
Transformer encoder / decoder layer chain
"""
import numpy as np
import tensorflow as tf

import lib.layers
from . import layers, ops
from .data import linelen


class Transformer:

    def __init__(
            self, name, inp_voc, out_voc,
            logits_bias=False, share_emb=False, dst_rand_offset=False,
            rescale_emb=True, inp_emb_bias=False, emb_inp_device='', emb_out_device='',
            **kwargs
    ):
        """
        Transformer-based model that predicts logp(insert(i, token) | x, y)
        :type inp_voc: lib.voc.Voc
        :type out_voc: lib.voc.Voc
        :param logits_bias: if True, final logits layer has bias term.
        :param share_emb: if True, input and output embeddings will use the same matrix.
            Useful for in case of shared vocabularies or when there is a
        :param dst_rand_offset: if True, adds a random offset to output embeddings, same for all positions
        :param kwargs: other hyperparameters - see TransformerChain and TransformerEmbedding
        """
        self.name = name
        self.inp_voc, self.out_voc = inp_voc, out_voc
        self.dst_rand_offset = dst_rand_offset
        self.hp = kwargs

        emb_size = kwargs.get('emb_size', kwargs.get('hid_size', 512))
        max_voc_size = max(len(inp_voc), len(out_voc))

        with tf.variable_scope(self.name) as self.scope:
            # Embeddings
            self.emb_inp = layers.TransformerEmbedding(
                'emb_inp', max_voc_size if share_emb else len(inp_voc), emb_size,
                bias=inp_emb_bias, rescale=rescale_emb, device=emb_inp_device)

            self.emb_out = layers.TransformerRelativeEmbedding(
                'emb_out', max_voc_size if share_emb else len(out_voc), emb_size,
                matrix=self.emb_inp.emb.mat if share_emb else None,
                rescale=rescale_emb, device=emb_out_device)

            # Model body
            self.encoder = layers.TransformerChain('enc', **kwargs)
            self.decoder = layers.TransformerChain('dec', attn_inputs=['enc'], **kwargs)

            # logits: token insertions plus one extra logit to predict position where to insert
            #NOTE(prkriley): specifying bias=0 and leaving activation unspecified means this is just a matrix mult
            #self.logits = layers.Dense(
            #    'logits', kwargs['hid_size'], len(out_voc) + 1,
            #    matrix=tf.transpose(self.emb_out.emb.mat) if kwargs.get('dwwt', False) else None,
            #    bias=None if logits_bias else 0
            #)
            assert not kwargs.get('dwwt', False)
            assert not logits_bias
            self.logits_W = layers.Dense('logits_W', kwargs['hid_size'], len(out_voc), matrix=None,bias=0)
            self.logits_D = layers.Dense('logits_D', kwargs['hid_size'], kwargs['hid_size'], matrix=None,bias=0)
            self.logits_E = layers.Dense('logits_E', kwargs['hid_size'], kwargs['hid_size'], matrix=None,bias=0)
            self.logits_F = layers.Dense('logits_F', kwargs['hid_size'], kwargs['hid_size'], matrix=None,bias=0)


    def _get_batch_sample(self):
        """ A minimal example of model input data """
        return [("i saw a cat", "i write the code")]

    def make_encoder_batch_ph(self):
        return {
            'inp': tf.placeholder('int32', [None, None]),
            'inp_len': tf.placeholder('int32', [None])
        }

    def make_feed_dict(self, batch, dummy_relative_positions=False, relative_positions_matrices=None, **kwargs):
        """ Take input data strings, return a dict { key: np.array(value) } """
        inp_lines, out_lines = zip(*batch)
        inp_len = [linelen(line) for line in inp_lines]
        out_len = [linelen(line) for line in out_lines]
        P = max(out_len) + 1
        if dummy_relative_positions:
            #TODO(prkriley): relative_positions has to actually be reasonable, in at least some cases
            relative_positions = np.ones([len(out_len),P,P],dtype=np.int32) # [batch, P, P]
            relative_positions[:,0,:] += 1 #not meaningful, just want to give variance to values
            relative_positions[:,-1,:] -= 1
        elif relative_positions_matrices is not None:
            relative_positions = []
            for i, m in enumerate(relative_positions_matrices):
                size = out_len[i] + 1
                assert m.shape == (size, size), "m.shape: {} =/= (size,size): {}".format(m.shape, (size,size))
                relative_positions.append(np.pad(m, ((0,P-size),(0,P-size))))

            relative_positions = np.stack(relative_positions, axis=0)
        else:
            relative_positions = None
        output = { 
            'inp': self.inp_voc.to_matrix(inp_lines),
            'inp_len': np.array(inp_len, 'int32'),
            'out': self.out_voc.to_matrix(out_lines),
            'out_len': np.array(out_len, 'int32'),}

        if relative_positions is not None:
            output['relative_positions'] = relative_positions
        return output

    def encode(self, batch, is_train):
        """ Take placeholders for data batch, return encoder state """
        with tf.name_scope(self.name), ops.dropout_scope(is_train):
            inp = batch['inp']  # [batch_size * ninp]
            inp_len = batch.get('inp_len', ops.infer_length(inp, self.inp_voc.eos))  # [batch]
            attn_mask = ops.make_attn_mask(inp, inp_len)  # [batch_size, 1, 1, ninp]
            out, _ = self.encoder(self.emb_inp(inp), self_attn_mask=attn_mask, relative_positions=None)
            # ^-- [batch_size, ninp, hid_size]
            return dict(out=out, attn_mask=attn_mask)

    def compute_action_logprobs(self, batch, is_train, enc=None, temperature=None, debug_inference=False):
        """
        Compute log-probabilities for all possible actions (aka agent policy)
        :param batch: a dict with
            - token matrix 'out'[batch_size, output_length]
            - optional length vector out_len[batch_size]
        :param is_train: whether or not to use training behavior (e.g. dropout)
        :returns: {'insert':logp(insert(i, c) | x, y), 'finish':logp(terminate| x, y)}
        """
        enc = self.encode(batch, is_train) if enc is None else enc
        with tf.name_scope(self.name), ops.dropout_scope(is_train):
            out = batch['out']  # partial translation, shape: [batch_size, nout]
            #NOTE(prkriley): above shape may actually be [batch_size, nout], just based on ops.infer_length, UNLESS out_len is already provided so infer_length isn't called...
            #NOTE(prkriley): no, it is CLEARLY 2D; look at out_padded
            if debug_inference:
                out = tf.Print(out, [tf.shape(out)], "out.shape: ", summarize=4)
            out_len = batch.get('out_len', ops.infer_length(out, self.out_voc.eos))  # [batch]

            # embedding. Note: at this point, a special "zero" vector is added
            # to the first position hence length is increased by 1

            out_padded = tf.concat([tf.zeros_like(out[:, :1]), out], axis=1)  # [batch_size, nout+1]
            dec_emb = self.emb_out(out_padded, offset='random' if self.dst_rand_offset else 0)
            # ^-- shape: [batch_size, nout + 1] #NOTE(prkriley): this may not be the right shape; surely there's a d_emb dim?; shoud be batch * ninp * emb_dim, where ninp is nout+1
            #NOTE(prkriley): emb_out's call method calls ops.make_transformer_timing_signal which will have to change with relative encodings

            # run decoder

            attn_mask = ops.make_causal_attn_mask(out_padded, out_len + 1)  # [batch_size, 1, nout + 1, nout + 1]
            #NOTE(prkriley): n_q is nout+1, inp_dim is emb_dim, n_kv is nout+1, output_depth is hid_size
            #TODO(prkriley): This needs to have the relative encoding stuff
            dec_out, _ = self.decoder(dec_emb, self_attn_mask=attn_mask,
                                      enc_out=enc['out'], enc_attn_mask=enc['attn_mask'], relative_positions=batch['relative_positions'])
            # ^-- [batch_size, nout + 1, hid_size]


            ##########
            #TODO(prkriley): compute new probabilities here

            #NOTE(prkriley): nout is number of real tokens + 1 for EOS, so nout+1 is Nreal+1EOS+1special
              #for non-special, indicates what to insert before
              #EOS is insert before EOS = at end
              #note that baseline always had EOS padded at end even for partial outputs; were there ever true EOSs?
              #Time dimension should be nout, not nout+1? Double-check
            #p(pos): [batch_size, time, nout + 1]
            #p(word|pos) : [batch_size, time, nout, voc_size]
            
            #NOTE(prkriley):  below:
            #P = nout + 1 represents slots to insert at plus one for terminating
            #P-1 = nout represents slots to insert given that we are not terminating
            #T = nout represents timesteps in the decoding process, which determines appropriate masking

            batch_size = tf.shape(dec_out)[0]
            P = tf.shape(dec_out)[1]
            T = P - 1
            H = dec_out # [batch_size, P, hid_size]
            H_prime = dec_out[:,:-1,:] # [batch_size, T, hid_size] or [batch_size, P-1, hid_size] depending on context
            if not is_train:
                H_prime = tf.batch_gather(H_prime, out_len[:, None]) # [batch_size, T=1, hid_size]
            position_selector = self.logits_D(H) # [batch_size, P, hid_size]
            timestep_selector_for_position = self.logits_E(H_prime) # [batch_size, T, hid_size]
            #TODO(prkriley): instead of -1, need to gather by actual last valid index
            #if not is_train:
                #timestep_selector_for_position = tf.batch_gather(timestep_selector_for_position, out_len[:, None]) # [batch_size, T=1, hid_size]
                #timestep_selector_for_position = timestep_selector_for_position[:,-1:,:] #only care about last timestep in inference
            position_logits = tf.matmul(timestep_selector_for_position, position_selector, transpose_b=True) # [batch_size, T, P]
            #TODO(prkriley): attention mask
            #at max T, same as baseline: extends into P dimension with out_len+1 Trues which is at most nout+1=P
            #BUT for each step back from T, one more True is gone, maximally T-1 gone, P-(T-1) = 2: insert into empty or terminate
              #NOTE that we have to make sure nothing wonky happens for the short sequences
            #how to construct? ignoring batch, start with range(P) and range(T)+1, broadcast both, do t >= p, then baseline mask
            #TODO(prkriley): is this actually any different from the causal attn_mask?
            time_position_mask = tf.greater_equal(tf.range(1, T+1)[None, :, None], tf.range(P)[None, None, :]) # [1, T, P]
            if not is_train:
                time_position_mask = time_position_mask[:,-1:,:] # This should be all True

            #NOTE(prkriley): after some point in dimension T, every row is just out_len True values
            time_position_mask = tf.logical_and(time_position_mask, tf.cast(attn_mask[:,0,-1:,:], tf.bool)) # [batch_size, T, P]
            position_logits = tf.where(time_position_mask, position_logits, tf.fill(tf.shape(position_logits), -1e9)) # [batch_size, T, P]
            position_logp = tf.nn.log_softmax(position_logits, axis=-1) # [batch_size, T, P]
            #TODO(prkriley): figure out finish_logp
                #before based on out_len, which now is a function of T: (out_len - (T-1-t))
                #out_len is at most P-1=T
                #T-1-t is just range(T) (but reversed, right?)
                #final position is fixed at first and grows through T, but caps at out_len
                    #starts at 1, max is min(T,out_len) = min(range(1, T+1), out_len)
            #finish_logp_indices = out_len[:, None] - tf.range(T-1, -1, -1)[None, :] # [batch_size, T]
            if is_train:
                finish_logp_indices = tf.minimum(tf.range(1,T+1)[None, :], out_len[:, None]) # [batch_size, T]
                finish_logp_indices = tf.stack([tf.broadcast_to(tf.range(batch_size)[:, None], [batch_size, T]), tf.broadcast_to(tf.range(T)[None, :], [batch_size, T]), finish_logp_indices], axis=-1) # [batch_size, T, 3]
            else:
                #NOTE(prkriley): tf.zeros because position_logp has already had the T dimension trimmed to the correct single value
                finish_logp_indices = tf.stack([tf.range(batch_size)[:, None], tf.zeros(tf.shape(out_len[:, None]), dtype=out_len.dtype), out_len[:, None]], axis=-1) # [batch_size, T=1, 3]
            finish_logp = tf.gather_nd(position_logp, finish_logp_indices) # [batch_size, T]

            insert_position_logp = tf.where(time_position_mask[:,:,1:], 
                                            position_logp[:,:,:-1], 
                                            tf.fill(tf.shape(position_logp[:,:,:-1]), -1e9)) # [batch_size, T, P-1]
            #TODO(prkriley): do token_logits: [batch_size, T, P-1, V]
            time_position_selector_for_tokens = self.logits_F(H_prime)[:,:,None,:] + position_selector[:,None,:-1,:] # [batch_size, T, P-1, hid_size]
            #if is_train:
                #pass
            #else:
                #time_position_selector_for_tokens = self.logits_F(H_prime[:,-1:,:])[:,:,None,:]
                #time_position_selector_for_tokens = time_position_selector_for_tokens + position_selector[:,None,:-1,:] # [batch_size, T, P-1, hid_size]
                #timestep_selector_for_position = tf.batch_gather(timestep_selector_for_position, out_len[:, None]) # [batch_size, T=1, hid_size]
            token_logits = self.logits_W(time_position_selector_for_tokens) # [batch_size, T, P-1, V]

            #token_logits = tf.Print(token_logits, [tf.shape(token_logits)], "token_logits.shape: ", summarize=4)
            token_logp_given_position = tf.nn.log_softmax(token_logits, axis=-1)

            insert_logp = insert_position_logp[:,:,:,None] + token_logp_given_position # [batch_size, T, P-1, V]
            
        return {
            'insert': insert_logp,  # [batch_size, T, nout, voc_size]
            'finish': finish_logp,  # [batch_size, T]
        }



            

        """
            ##########

            #TODO(prkriley): fix logits
            logits = self.logits(dec_out)  # [batch_size, nout + 1, voc_size + 1] #NOTE(prkriley): for refactor, likely need another nout dimension (maybe + 1?) which conceptually is the timestep dim
            if temperature is not None:
                logits /= temperature

            # compute log-probabilities for actions

            # position log-probabilities, logP(insert(pos, *) | ...)
            # used to predict position of next insert and termination condition (EOS)
            position_logits = logits[:, :, -1]  # [batch_size, nout + 1]

            position_mask = tf.cast(attn_mask, tf.bool)[:, 0, 0, :]  # [batch_size, nout + 1]
            position_logits = tf.where(position_mask, position_logits,
                                       tf.fill(tf.shape(position_logits), -1e9))
            position_logp = tf.nn.log_softmax(position_logits, axis=-1)  # [batch_size, n_out] #NOTE(prkriley): n_out+1

            # two actions: insert - at any non-EOS position - or finish - defined as inserting at EOS
            finish_logp = tf.gather_nd(position_logp,
                                       tf.stack([tf.range(tf.shape(out_len)[0]), out_len], axis=1))
            # ^-- [batch_size]

            insert_position_logp = tf.where(position_mask[:, 1:], position_logp[:, :-1],
                                            tf.fill(tf.shape(position_logp[:, :-1]), -1e9))
            # ^-- [batch_size, nout]

            # insertion log-probabilities:
            # logP(insert(pos, tok) | ...) = logP(insert(pos, *) | ...) + logP(insert(pos, tok) | insert(pos, *), ...)

            token_logits = logits[:, :-1, :len(self.out_voc)]  # [batch_size, n_out, voc_size]
            token_logp_given_position = tf.nn.log_softmax(token_logits, axis=-1)
            # note: we do not need mask on token_logp_given_position cuz mask is already applied to insert_position_logp

            insert_logp = insert_position_logp[:, :, None] + token_logp_given_position

        return {
            # group 1 (exps sum to 1)
            'insert': insert_logp,  # [batch_size, nout, voc_size]
            'finish': finish_logp,  # [batch_size]
        }
        """


class ImgToSeqTransformer(Transformer):
    def __init__(
            self, name, out_voc, inp_w, inp_h, inp_channels=3, make_encoder=lib.layers.ImageEncoder,
            logits_bias=False, share_emb=False, dst_rand_offset=False,
            rescale_emb=True, emb_out_device='',
            **kwargs
    ):
        """
        Transformer-based model that predicts logp(insert(i, token) | x, y)
        :type out_voc: lib.voc.Voc
        :param logits_bias: if True, final logits layer has bias term.
        :param dst_rand_offset: if True, adds a random offset to output embeddings, same for all positions
        :param kwargs: other hyperparameters - see TransformerChain and TransformerEmbedding
        """
        self.name = name
        self.inp_voc, self.out_voc = out_voc, out_voc  # inp voc is a stub, the same as out_voc
        self.dst_rand_offset = dst_rand_offset
        self.hp = kwargs
        self.w = inp_w
        self.h = inp_h
        self.inp_channels = inp_channels

        emb_size = kwargs.get('emb_size', kwargs.get('hid_size', 512))
        max_voc_size = len(out_voc)

        with tf.variable_scope(self.name) as self.scope:
            # Embeddings

            self.emb_out = layers.TransformerEmbedding(
                'emb_out', max_voc_size if share_emb else len(out_voc), emb_size,
                matrix=self.emb_inp.emb.mat if share_emb else None,
                rescale=rescale_emb, device=emb_out_device)

            # Model body
            self.encoder = make_encoder('enc', inp_h=inp_w, inp_w=inp_h, inp_channels=inp_channels, **kwargs)

            enc_out_shape = self.encode(self.make_encoder_batch_ph(), True)['out'].shape
            assert enc_out_shape.ndims == 3 and enc_out_shape[-1].value is not None, \
                "encoder output shape must be a 3d tensor with fixed num units, " \
                "got shape {}".format(enc_out_shape)

            self.decoder = layers.TransformerChain('dec', attn_inputs=['enc'],
                                                   attn_input_sizes={'enc': enc_out_shape[-1].value},
                                                   **kwargs)

            # logits: token insertions plus one extra logit to predict position where to insert
            self.logits = layers.Dense(
                'logits', kwargs['hid_size'], len(out_voc) + 1,
                bias=None if logits_bias else 0
            )


    def _get_batch_sample(self):
        """ A minimal example of model input data """
        return [(np.zeros((self.h, self.w, self.inp_channels)), 'A cat sat')]

    def make_feed_dict(self, batch, **kwargs):
        """ Take input data strings, return a dict { key: np.array(value) } """
        inp_imgs, out_lines = zip(*batch)

        out_len = [linelen(line) for line in out_lines]
        return {
            'inp': np.array(inp_imgs, 'float32'),
            'out': self.out_voc.to_matrix(out_lines),
            'out_len': np.array(out_len, 'int32')
        }

    def make_encoder_batch_ph(self):
        return {
            'inp': tf.placeholder('float32', [None, self.h, self.w, self.inp_channels]),
        }

    def encode(self, batch, is_train):
        """ Take placeholders for data batch, return encoder state """
        with tf.name_scope(self.name), ops.dropout_scope(is_train):
            inp = batch['inp']  # [batch_size * ninp]

            out = self.encoder(inp)
            assert out.shape[-1] is not None
            out_shape = tf.shape(out)

            out = tf.reshape(out, [out_shape[0], -1, out.shape[-1]])

            attn_mask = tf.ones((out_shape[0], 1, 1, out_shape[1] * out_shape[2]))  # [batch_size, 1, 1, ninp]

            return dict(out=out, attn_mask=attn_mask)
