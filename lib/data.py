"""Calibration / perplexity data loaders.

Calibration corpora: WikiText-2, PTB, C4 (streaming).
Following LinearPatch (https://github.com/chenxinrui-tsinghua/LinearPatch), the
default calibration set is 128 sequences of length 2,048 from the C4 train split.
"""
from datasets import load_dataset
import torch
import torch.nn as nn
import random
from tqdm import tqdm


def get_wikitext2(tokenizer, train_size, val_size, seed, seqlen, test_only):
    print("get wikitext2")
    testdata = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    testenc = tokenizer("\n\n".join(testdata["text"]), return_tensors="pt")
    if test_only:
        return testenc

    traindata = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    trainenc = tokenizer("\n\n".join(traindata["text"]), return_tensors="pt")

    random.seed(seed)
    trainloader = []
    val_sample_ratio = 0.9  # train from [0:0.9], val from [0.9:1.0] (no overlap)
    for _ in range(train_size):
        i = random.randint(0, int(trainenc.input_ids.shape[1] * val_sample_ratio) - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    valloader = []
    for _ in range(val_size):
        i = random.randint(int(trainenc.input_ids.shape[1] * val_sample_ratio) - seqlen - 1,
                           trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        valloader.append((inp, tar))
    return trainloader, valloader


def get_ptb(tokenizer, train_size, val_size, seed, seqlen, test_only):
    print("get_ptb")
    valdata = load_dataset("ptb_text_only", "penn_treebank", split="validation",
                           trust_remote_code=True)
    testenc = tokenizer("\n\n".join(valdata["sentence"]), return_tensors="pt")
    if test_only:
        return testenc

    traindata = load_dataset("ptb_text_only", "penn_treebank", split="train",
                             trust_remote_code=True)
    trainenc = tokenizer("\n\n".join(traindata["sentence"]), return_tensors="pt")

    random.seed(seed)
    trainloader = []
    for _ in range(train_size):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc


def get_c4(tokenizer, train_size, val_size, seed, seqlen, test_only):
    print("get_c4")
    traindata = load_dataset("allenai/c4", "en", split="train",
                             streaming=True).shuffle(seed=seed, buffer_size=10_000)
    validationdata = load_dataset("allenai/c4", "en", split="validation",
                                  streaming=True).shuffle(seed=seed, buffer_size=10_000)
    train_it = iter(traindata)
    val_it = iter(validationdata)

    def next_window(it, rng):
        """Sample one window (inp, tar) of length seqlen from the stream."""
        while True:
            ex = next(it)
            enc = tokenizer(ex["text"], return_tensors="pt")
            L = enc.input_ids.shape[1]
            if L >= seqlen + 1:
                start = rng.randint(0, L - seqlen - 1)
                end = start + seqlen
                inp = enc.input_ids[:, start:end]
                tar = inp.clone()
                tar[:, :-1] = -100
                return inp, tar

    rng_valenc = random.Random(0)
    valenc = []
    for _ in range(256):
        inp, _ = next_window(val_it, rng_valenc)
        valenc.append(inp)
    valenc = torch.hstack(valenc)  # [1, 256*seqlen]
    if test_only:
        return valenc

    rng_train = random.Random(seed)
    trainloader = []
    for _ in range(train_size):
        inp, tar = next_window(train_it, rng_train)
        trainloader.append((inp, tar))

    rng_val = random.Random(seed + 1)
    valloader = []
    for _ in range(val_size):
        inp, tar = next_window(val_it, rng_val)
        valloader.append((inp, tar))

    return trainloader, valloader


def get_loaders(name, tokenizer, train_size=128, val_size=16, seed=0,
                seqlen=2048, test_only=False):
    if "wikitext2" in name:
        return get_wikitext2(tokenizer, train_size, val_size, seed, seqlen, test_only)
    elif "c4" in name:
        return get_c4(tokenizer, train_size, val_size, seed, seqlen, test_only)
    elif "ptb" in name:
        return get_ptb(tokenizer, train_size, val_size, seed, seqlen, test_only)
    else:
        raise NotImplementedError(name)


@torch.no_grad()
def test_ppl(model, tokenizer, datasets=("wikitext2",), ppl_seqlen=2048):
    """Perplexity on non-overlapping windows of length ppl_seqlen."""
    results = {}
    for dataset in datasets:
        testloader = get_loaders(dataset, tokenizer, seed=0, seqlen=ppl_seqlen,
                                 test_only=True)
        testenc = testloader if "c4" in dataset else testloader.input_ids

        seqlen = ppl_seqlen
        nsamples = testenc.numel() // seqlen
        nlls = []
        classifier = model.lm_head
        for i in tqdm(range(nsamples)):
            batch = testenc[:, (i * seqlen):((i + 1) * seqlen)].to(model.device)
            outputs = model.model(batch)
            hidden_states = outputs[0]
            logits = classifier(hidden_states.to(classifier.weight.dtype))
            shift_logits = logits[:, :-1, :]
            shift_labels = testenc[:, (i * seqlen):((i + 1) * seqlen)][:, 1:].to(shift_logits.device)
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)),
                            shift_labels.view(-1))
            nlls.append(loss.float() * seqlen)

        ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * seqlen))
        results[dataset] = ppl.item()
    return results
