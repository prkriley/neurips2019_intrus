#!/usr/bin/env python
# coding: utf-8

# __this notebook__ pre-trains INTRUS transformer transformer on uniform random trajectories. It requires the same data as the pretrain notebook plus three files:

# In[2]:


import os, sys
sys.path.append('..')
import numpy as np
import tensorflow as tf
tf.reset_default_graph()
sess = tf.InteractiveSession(config=tf.ConfigProto(allow_soft_placement=True))

experiment_name = "enru_full_pretrain_v1_100k_L"
DATA = "" #or _toy
# !rm -rf {experiment_name}
assert not os.path.exists(experiment_name), "please use unique name for each experiment"


# ### Data preprocessing

# In[3]:


from prefetch_generator import background
from lib.data import form_batches, form_adaptive_batches_windowed, filter_by_len, cycle_shuffle, maxlen, linelen
from lib.multi_gpu import ParallelBatchIterator

class train:
    inp_lines = list(open('../data/training/train.en{}.tok.bpe'.format(DATA)))
    out_lines = list(open('../data/training/train.ru{}.tok.bpe'.format(DATA)))
    
    batcher = background(max_prefetch=256)(form_adaptive_batches_windowed)(
        filter_by_len(cycle_shuffle(zip(inp_lines, out_lines)), max_srclen=200, max_dstlen=200),
        max_size=2048, batch_size_max=256, split_len=10000,
    )
    parallel_batcher = background(max_prefetch=256)(ParallelBatchIterator)(batcher, 
                                     cost_func=lambda batch: max(linelen(pair[1]) for pair in batch) ** 2,
                                     n_buffers=1) #NOTE(prkriley): I think n_buffers corresponds to gpu number; was 8
    
class dev:
    inp_lines = list(open('../data/dev/dev.en{}.tok.bpe'.format(DATA)))
    out_lines = list(open('../data/dev/dev.ru{}.tok.bpe'.format(DATA)))
    
    #NOTE(prkriley): 256 is weird, should just be all of them...
    batcher = background(max_prefetch=256)(form_batches)(
        cycle_shuffle(zip(inp_lines, out_lines)),
        batch_size=256
    )


# ### Model & vocabs

# In[4]:


from lib.voc import Voc
inp_voc = Voc.from_sequences(train.inp_lines)
out_voc = Voc.from_sequences(train.out_lines)


# In[5]:


from lib.models import Transformer
from lib.trainer import FixedOrderTrainer
from lib.multi_gpu import MultiGPUTrainer
from lib.saveload import save, load

hp = {
    'emb_size': 512,
    'hid_size': 512,
    'num_heads': 8,
    'ff_size': 2048,
    'num_layers': 6,
    'rescale_emb': True,
    'inp_emb_bias': False,
    'res_steps': 'nlda',
    'normalize_out': True,
    'attn_dropout': 0.0,
    'res_dropout': 0.1,
    'relu_dropout': 0.0,
    'share_emb': False,
}

make_model = lambda: Transformer('mod', inp_voc, out_voc, **hp)
trainer = MultiGPUTrainer('trainer', make_model,
                          TrainerClass=FixedOrderTrainer, sampler_opts=dict(samples_per_line=1),
                          optimizer_opts=dict(base_lr=1.4e-3, warmup_time=16000))

sess.run(tf.global_variables_initializer())


# In[8]:


import nltk
from lib.inference import BeamSearchInserts
model = trainer.master_model
decoder = BeamSearchInserts(model)

def compute_bleu(batch, beam_size=32, beam_spread=50, len_alpha=1.0, 
                 maxlen=lambda inp_line: len(inp_line.split()) * 2 + 3, unbpe=False, log_f=None):
    """
    Computes corpora-level bleu on batch.
    Note: this isn't exactly the same BLEU as what's used for model evaluation, but very close to one.
        Use moses bleu for final evaluation.
    """
    translations, references = [], []
    for inp, ref_out in batch:
        hypo_scores = decoder.translate_line(inp, beam_size=beam_size, beam_spread=beam_spread,
                                             max_steps=maxlen(inp))
        hypo_scores = decoder.apply_length_penalty(hypo_scores, len_alpha=len_alpha)
        best_trans = max(hypo_scores.keys(), key=hypo_scores.get)
        if unbpe:
            best_trans = best_trans.replace('@@ ', '').replace(' `', '')
            ref_out = ref_out.replace('@@ ', '').replace(' `', '')
            
        if log_f:
          log_f.write('{}\t{}'.format(best_trans,ref_out))
        translations.append(best_trans)
        references.append(ref_out)
    
    return nltk.bleu_score.corpus_bleu(
        [[ref.split()] for ref in references], 
        [trans.split() for trans in translations])


# In[9]:


hypo_scores = decoder.translate_line('i am the monument to all your sins .',
                                     beam_size=4, beam_spread=10)
hypo_scores = decoder.apply_length_penalty(hypo_scores, len_alpha=1.5)
max(hypo_scores.keys(), key=hypo_scores.get)


# ### Training

# In[10]:


#from IPython.display import clear_output
#import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')
from tensorboardX import SummaryWriter
import pandas as pd
ewma = lambda x, span: pd.DataFrame({'x': x})['x'].ewm(span=span).mean().values

loss_history, acc_history, dev_bleu_history = [], [], []
writer = SummaryWriter(log_dir=experiment_name)


# In[12]:


from itertools import chain
from tqdm import trange

#for t in trange(100000):
for t in trange(100000): #NOTE(prkriley): 100k for first attempt on toy (10%) data, didn't finish (got like 57k in 5 days on -g -S); from 25k to 1k for first attempt for full data
    batches = chain(*[next(train.parallel_batcher) for _ in range(4)])
    metrics_t = trainer.train_on_batches(batches, slice_max_len=4600, optimizer_step=True)
    step = metrics_t['step']
    
    for key in metrics_t:
        writer.add_scalar(key, metrics_t[key], global_step=metrics_t['step'])
    loss_history.append(metrics_t['loss'])
    acc_history.append(metrics_t['acc'])
    
    if step % 100 == 0:
        if step % 2000 == 0:
          log_f = open('{}/step_{}.hypref'.format(experiment_name,step), 'w')
        else:
          log_f = None
        dev_bleu_t = compute_bleu(next(dev.batcher), unbpe=True, log_f=log_f)
        if log_f:
          log_f.close()
        dev_bleu_history.append([len(loss_history), dev_bleu_t])
        writer.add_scalar('dev_BLEU', dev_bleu_t, global_step=step)
    
    if step % 10000 == 0:
        save(os.path.join(experiment_name, 'checkpoint_%i.npz' % step),
             [var for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES) if 
              var.name.startswith('trainer/worker_0') or not var.name.startswith('trainer/worker')])
        
    if step % 100 == 0 and False: #NOTE(prkriley): I added the and False to skip plotting
        clear_output(True)
        plt.figure(figsize=[18, 6])
        plt.subplot(1, 3, 1)
        plt.scatter(np.arange(len(loss_history)), loss_history, alpha=0.1)
        plt.plot(ewma(loss_history, span=100), color='orange')
        plt.title('train loss'); plt.grid()
        
        plt.subplot(1, 3, 2)
        plt.scatter(np.arange(len(acc_history)), acc_history, alpha=0.1)
        plt.plot(ewma(acc_history, span=100), color='orange')
        plt.title('train acc'); plt.grid()
        
        plt.subplot(1, 3, 3)
        dev_bleu_steps, dev_bleu_values = zip(*dev_bleu_history)
        plt.scatter(dev_bleu_steps, dev_bleu_values, alpha=0.1)
        plt.plot(dev_bleu_steps, ewma(dev_bleu_values, span=10), color='orange')
        plt.title('dev bleu'); plt.grid();
        plt.show()


# In[15]:


#decoder.translate_line('i am the monument to all your sins .', max_steps=20)


# In[ ]:


save(os.path.join(experiment_name, 'checkpoint_final.npz'), tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))
inp_voc.save('./{}/inp.voc'.format(experiment_name))
out_voc.save('./{}/out.voc'.format(experiment_name))

print(os.path.join(experiment_name, 'checkpoint_%i.npz' % step))

