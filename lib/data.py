import random


def linelen(line):
    "num tokens + 1 for eos"
    return len(line.split()) + 1


def maxlen(item):
    return max(linelen(l) for l in item if isinstance(l, str))


def cycle_shuffle(iterator):
    items = list(iterator)
    while True:
        random.shuffle(items)
        yield from items


def form_batches(data, batch_size):
    seq = iter(data)
    done = False
    while not done:
        batch = []
        for _ in range(batch_size):
            try:
                batch.append(next(seq))
            except StopIteration:
                done = True
        if batch:
            yield batch


def form_adaptive_batches_grouped(data, batch_cost, cost_func=maxlen, batch_size_max=0):
    from collections import defaultdict
    import sys
    #TODO(prkriley): implement
    """
    iterate, bin by id, calculate bin total, perform similar logic to non-grouped but over bins
    need to figure out field names, and make sure they are real and not Tensors
    field is: out_to_inp_indices, it's one number (name because grouped later), not Tensor
    """
    #bin by index
    import math
    bins = defaultdict(list)
    max_lens = defaultdict(int)
    for item in data:
      idx = item['out_to_inp_indices']
      if not batch_size_max or len(bins[idx]) < batch_size_max:
        bins[idx].append(item)
        max_lens[idx] = max(max_lens[idx], cost_func(item))
      elif batch_size_max:
        print('Warning: Single hypothesis sequence longer than specified max size of {}! Truncating.'.format(batch_size_max), file=sys.stderr)

    #truncation
    for idx in list(bins.keys()):
      if len(bins[idx]) * max_lens[idx] > batch_cost:
        cummaxes = [0]
        cummax = 0
        for e in bins[idx]:
          cummax = max(cummax, cost_func(e))
          cummaxes.append(cummax)
        trunc_costs = [i*cummaxes[i] for i in range(len(bins[idx]) + 1)]
        trunc_idx = len(trunc_costs) - 1
        while trunc_idx > 0 and trunc_costs[trunc_idx] > batch_cost:
          trunc_idx -= 1
        if trunc_idx > 0:
          bins[idx] = bins[idx][:trunc_idx]
          max_lens[idx] = cummaxes[trunc_idx]
        else:
          del bins[idx]


    
    slices = []
    this_slice = []
    max_len = 0
    total_len = 0
    for idx, items in bins.items():
      max_len = max(max_len, max_lens[idx])
      if (total_len + len(items)) * max_len > batch_cost or (batch_size_max and (total_len + len(items)) > batch_size_max):
        assert len(this_slice) > 0, 'Bug in truncation code; first truncated hypothesis still too big! len {}, max_len {}.'.format(len(items),max_len)
        slices.append(this_slice)
        this_slice = [idx]
        max_len = max_lens[idx]
        total_len = len(items)
      else:
        this_slice.append(idx)
        total_len += len(items)
    #end for
    slices.append(this_slice)
    for indices in slices:
      slice_len = 0
      slice_max_len = 0
      s = []
      for idx in indices:
        s += bins[idx]
        slice_max_len = max(slice_max_len, max_lens[idx])
        slice_len += len(bins[idx])
      print('Slice len, max_len, total_cost: {}, {}, {}'.format(slice_len, slice_max_len, slice_len*slice_max_len))
      yield s


def form_adaptive_batches(data, batch_cost, cost_func=maxlen, batch_size_max=0):
    seq = iter(data)
    prev = []
    max_len = 0
    done = False
    while not done:
        batch = prev
        try:
            while True:
                item = next(seq)
                max_len = max(max_len, cost_func(item))
                if (len(batch) + 1) * max_len > batch_cost or (batch_size_max and len(batch) >= batch_size_max):
                    prev, max_len = [item], cost_func(item)
                    break
                batch.append(item)
        except StopIteration:
            done = True
        if batch:
            yield batch


def form_adaptive_batches_windowed(data, cost_func=maxlen, sort_key_func=None,
                                   max_size=5000, split_len=10000, batch_size_max=0):
    rng = random.Random(42)
    buf = []
    last_chunk = []
    reverse = False
    for p in data:
        if len(buf) >= split_len:
            # Last chunk may contain fewer sentences than others - let's return in to the miller
            buf += last_chunk

            buf = sorted(buf, key=sort_key_func or cost_func, reverse=reverse)
            chunks = list(form_adaptive_batches(buf, max_size, cost_func=cost_func, batch_size_max=batch_size_max))

            last_chunk = chunks.pop()
            buf = []

            reverse = not reverse

            rng.shuffle(chunks)
            for chunk in chunks:
                yield chunk
        buf.append(p)

    buf += last_chunk
    buf = sorted(buf, key=sort_key_func or cost_func, reverse=reverse)
    chunks = list(form_adaptive_batches(buf, max_size, cost_func=cost_func, batch_size_max=batch_size_max))
    rng.shuffle(chunks)
    for chunk in chunks:
        yield chunk


def filter_by_len(data, max_srclen=None, max_dstlen=None, batch_len=None):
    def item_ok(item):
        return ((max_srclen is None or linelen(item[0]) <= max_srclen) and
                (max_dstlen is None or linelen(item[0]) <= max_dstlen) and
                (batch_len is None or maxlen(item) <= batch_len))
    return filter(item_ok, data)
