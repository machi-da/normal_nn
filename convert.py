from chainer.dataset.convert import to_device


def convert(batches, gpu_id=None):
    src = []
    trg_sos = []
    trg_eos = []
    label = []
    for batch in batches:
        src.append([to_device(gpu_id, b) for b in batch[0]])
        trg_sos.append(batch[1])
        trg_eos.append(batch[2])
        label.append(batch[3])
    return src, trg_sos, trg_eos, label