import argparse
from config import Config
import torch
import torch.nn as nn
import torch.optim as optim
import random
import math
import time
from tqdm import tqdm
from torchtext.data import Iterator, BucketIterator

from model import Encoder, Decoder, Attention, Seq2Seq, Encoder_with_SelfAttn
from utils import *


def train(model, src_field, trg_field, iterator, optimizer, criterion, clip, teacher_force):
    model.train()
    epoch_loss = 0
    for i, batch in enumerate(iterator):
        src = getattr(batch, src_field)
        trg = getattr(batch, trg_field)
        optimizer.zero_grad()
        output = model(src, trg, teacher_force)
        # trg = [trg sent len, batch size]
        # output = [trg sent len, batch size, output dim]
        output = output[1:].view(-1, output.shape[-1])
        trg = trg[1:].view(-1)
        # trg = [(trg sent len - 1) * batch size]
        # output = [(trg sent len - 1) * batch size, output dim]
        loss = criterion(output, trg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(iterator)


def evaluate(model, src_field, trg_field, iterator, criterion):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            src = getattr(batch, src_field)
            trg = getattr(batch, trg_field)
            output = model(src, trg, 0)  # turn off teacher forcing
            # trg = [trg sent len, batch size]
            # output = [trg sent len, batch size, output dim]
            output = output[1:].view(-1, output.shape[-1])
            trg = trg[1:].view(-1)
            # trg = [(trg sent len - 1) * batch size]
            # output = [(trg sent len - 1) * batch size, output dim]
            loss = criterion(output, trg)
            epoch_loss += loss.item()
    return epoch_loss / len(iterator)


def decode_story(output, vocab, join=' '):
    """Converts a sequence of word ids to a story"""
    # TODO: Without repeat in each sentence
    id_tensor = output.argmax(2)
    ids = id_tensor.transpose(0, 1)
    sent_batch = []
    for i in range(ids.shape[0]):
        sent = []
        for j in range(ids.shape[1]):
            w = vocab.itos[ids[i][j]]
            sent.append(w)
        sent = join.join(sent)
        sent_batch.append(sent)
    return sent_batch


def decode_story_line(output, vocab, join=' '):
    """Converts a sequence of word ids to a story-line (Without repeat)"""
    # shape [trg sent len(7), batch size(1), output dim]
    output = output.squeeze(1)
    sent = []
    values, indices = torch.topk(output, k=7, dim=1)
    decoded_indices = []
    # forbid any word to appear twice in storyline
    for i in range(output.shape[0]):  # for the i-th word
        for idx in indices[i]:  # for top-k candidate
            if idx not in decoded_indices:
                # a new word
                w = vocab.itos[idx]
                decoded_indices.append(idx)
                if w == '<unk>':  #
                    continue
                sent.append(w)
                break
    sent = join.join(sent)
    return [sent]


def test_generate(model, src_field, trg_field, iterator, criterion, result_path, decode_func, compute_loss):
    model.eval()
    epoch_loss = 0
    generated_sentence = []
    with torch.no_grad():
        for i, batch in tqdm(enumerate(iterator)):
            src = getattr(batch, src_field)
            trg = getattr(batch, trg_field) if compute_loss else None
            output = model(src, trg, 0)  # turn off teacher forcing
            # trg = [trg sent len, batch size]
            # output = [trg sent len, batch size, output dim]
            generated_sentence.extend(decode_func(output, VOCAB.vocab, join=' '))
            output = output[1:].view(-1, output.shape[-1])
            if compute_loss:
                trg = trg[1:].view(-1)
                # trg = [(trg sent len - 1) * batch size]
                # output = [(trg sent len - 1) * batch size, output dim]
                loss = criterion(output, trg)
                epoch_loss += loss.item()
    test_loss = epoch_loss / len(iterator) if epoch_loss else 'Test Loss Not Computed.'
    with open(result_path, 'w') as f:
        for sent in generated_sentence:
            f.write(sent + '\n')
    return test_loss, generated_sentence


if __name__ == '__main__':
    # parse args
    parser = argparse.ArgumentParser('Plan and write')
    parser.add_argument('--config', default='Title2Line',
                        help='For code cleanliness, this is a pointer to config in config.py, '
                             'where specific config should be set')
    parser.add_argument('--mode', default='train_generate', choices=['train', 'generate', 'train_generate'])
    args = parser.parse_args()

    CONFIG = getattr(Config, args.config)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device:{device}')

    # set random seed
    SEED = 1234
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

    # dataset
    trainset_path = CONFIG['trainset']
    validset_path = CONFIG['validset']
    testset_path = CONFIG['testset']
    print(f'train/valid/test dataset path:{trainset_path}/{validset_path}/{testset_path}')

    src_field, trg_field = CONFIG['src_field'], CONFIG['trg_field']
    named_fields = [(src_field, VOCAB), (trg_field, VOCAB)]
    print(f'src_filed:{src_field}, trg_field:{trg_field}')
    DataSet = TitleLine if trg_field == 'story_line' else LineStory
    train_data = DataSet(path=trainset_path, format='tsv', fields=named_fields)
    valid_data = DataSet(path=validset_path, format='tsv', fields=named_fields)
    test_data = DataSet(path=testset_path, format='tsv', fields=named_fields)
    print(f"Number of training examples: {len(train_data.examples)}")
    print(f"Number of validation examples: {len(valid_data.examples)}")
    print(f"Number of testing examples: {len(test_data.examples)}")
    print('Show data example')
    print(vars(train_data.examples[0]), vars(valid_data.examples[0]), vars(test_data.examples[0]))

    # build vocabulary
    VOCAB.build_vocab(train_data, min_freq=CONFIG['min_freq'])
    print(f"Unique tokens in vocabulary(min frequency:{CONFIG['min_freq']}): {len(VOCAB.vocab)}")

    # data loader
    train_iterator, valid_iterator = BucketIterator.splits((train_data, valid_data),
                                                           batch_size=CONFIG['batch_size'], device=device)
    test_iterator = Iterator(test_data, batch_size=1, device=device, shuffle=False)

    # model
    INPUT_DIM = len(VOCAB.vocab)
    OUTPUT_DIM = len(VOCAB.vocab)
    ENC_EMB_DIM = CONFIG['enc_emb_dim']
    DEC_EMB_DIM = CONFIG['dec_emb_dim']
    ENC_HID_DIM = CONFIG['enc_hid_dim']
    DEC_HID_DIM = CONFIG['dec_hid_dim']
    N_LAYERS = CONFIG['n_layers']
    ENC_DROPOUT = CONFIG['enc_dropout']
    DEC_DROPOUT = CONFIG['dec_dropout']
    attn = Attention(ENC_HID_DIM, DEC_HID_DIM)
    # enc = Encoder(INPUT_DIM, ENC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, ENC_DROPOUT)
    enc = Encoder_with_SelfAttn(INPUT_DIM, ENC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, ENC_DROPOUT)
    dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, DEC_DROPOUT, attn)
    model = Seq2Seq(enc, dec, device).to(device)
    print(f'The model has {count_parameters(model)} trainable parameters')

    optimizer = optim.Adam(model.parameters())
    PAD_IDX = VOCAB.vocab.stoi['<pad>']
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    N_EPOCHS = CONFIG['n_epoch']
    CLIP = CONFIG['clip_norm']
    TEACHER_FORCE = CONFIG['teacher_force']
    MODEL_PATH = CONFIG['model_path']

    # train & valid
    if 'train' in args.mode:
        print(f'Training {src_field} to {trg_field} model')
        best_valid_loss = float('inf')
        for epoch in range(N_EPOCHS):
            start_time = time.time()
            train_loss = train(model, src_field, trg_field, train_iterator, optimizer, criterion, CLIP, TEACHER_FORCE)
            valid_loss = evaluate(model, src_field, trg_field, valid_iterator, criterion)
            end_time = time.time()
            epoch_mins, epoch_secs = epoch_time(start_time, end_time)
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                torch.save(model.state_dict(), MODEL_PATH)
            print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
            print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
            print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')

    # test & generate
    if 'generate' in args.mode:
        model.load_state_dict(torch.load(MODEL_PATH))
        RESULT_PATH = CONFIG['result_path']
        decode_func = decode_story_line if trg_field == 'story_line' else decode_story
        compute_loss = True if trg_field == 'story_line' else False
        test_loss, result = test_generate(model, src_field, trg_field, test_iterator, criterion, RESULT_PATH,
                                          decode_func, compute_loss)
        if trg_field == 'story_line':
            print(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss)} |')
            create_testfile(RESULT_PATH)
        elif trg_field == 'story':
            test_bleu(RESULT_PATH)
