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

def decode_story(output, vocab, join=' '):
    """Converts a sequence of word ids to a story"""
    # TODO: Without repeat in each sentence
    # shape [trg sent len(7), batch size(1), output dim]
    output = output.squeeze(1)
    K = 10
    values, indices = torch.topk(output, k=K, dim=1)
    sent = []
    decoded_indices = []
    # forbid any word to appear twice in storyline
    for i in range(output.shape[0]):  # for the i-th word
        for idx in indices[i]:  # for top-k candidate
            if idx not in decoded_indices:
                # a new word
                w = vocab.itos[idx]
                if w == '<eos>':
                    decoded_indices = []
                else:
                    decoded_indices.append(idx)
                    if len(decoded_indices) == K:
                        decoded_indices = []
                sent.append(w)
                break
    sent = join.join(sent)
    return [sent]


