import pickle, random
from collections import defaultdict

tag2label = {'O': 0, 'B-IPT': 1, 'I-IPT': 2}


def build_vocabulary(vocab_path, corpus_path, min_count):
    '''
    Build vocabulary for characters
    :return: A dictionary
    '''
    data = read_corpus(corpus_path)
    word2idx = defaultdict(int)
    for sent_, tag_ in data:
        for word in sent_:
            if word.isdigit():
                word = '<NUM>'
            elif ('\u0041' <= word <= '\u005a') or ('\u0061' <= word <= '\u007a'):
                word = '<ENG>'
            word2idx[word] += 1
    stop_words = []
    for word, count in word2idx.items():
        if count < min_count and word != '<NUM>' and word != '<ENG>':
            stop_words.append(word)
    for word in stop_words:
        word2idx.pop(word)

    # rerank vocabulary
    new_id = 1
    for word in word2idx.keys():
        word2idx[word] = new_id
        new_id += 1
    word2idx['<PAD>'] = 0
    word2idx['<UNK>'] = new_id
    print(len(word2idx))

    with open('./dataset/vocabulary.txt', 'w', encoding='utf-8') as f:
        for k, v in word2idx.items():
            f.write("{}\t{}\n".format(k, v))

    with open(vocab_path, 'wb') as f:
        pickle.dump(word2idx, f)


def read_corpus(path):
    '''
     Load corpus and return a list of samples
     Return: [([sent1, sent2, sent3 ...], [tag1, tag2, tag3 ...]), ...]
    '''
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        sent_, tag_ = [], []
        for line in f:
            if line != '\n':
                line = line.strip('\n')
                sent, tag = line.split(' ')
                sent_.append(sent)
                tag_.append(tag)
            else:
                data.append((sent_, tag_))
                sent_, tag_ = [], []

    return data


def load_pretrain_embedding(model):
    '''
    Load pretrain word_embedding matrix
    '''
    pass


def load_vocabulary(path):
    '''
    Load word vocabulary with dictionary format
    '''
    with open(path, 'rb') as f:
        word2idx = pickle.load(f)
    print('vocabulary_size : {}'.format(len(word2idx)))
    return word2idx


def sentence2id(sent, word2id):
    '''
    Word tokenizer for one sentence
    '''
    sentence_id = []
    for word in sent:
        if word.isdigit():
            word = '<NUM>'
        elif ('\u0041' <= word <= '\u005a') or ('\u0061' <= word <= '\u007a'):
            word = '<ENG>'
        if word not in word2id:
            word = '<UNK>'
        sentence_id.append(word2id[word])
    return sentence_id


def batch_yield(data, batch_size, vocab, tag2label, shuffle=False):
    '''
    Data generator
    Return : one batch dataset
    '''
    if shuffle:
        random.shuffle(data)

    seqs, labels = [], []
    for (sent_, tag_) in data:
        sent_ = sentence2id(sent_, vocab)
        label_ = [tag2label[tag] for tag in tag_]

        if len(seqs) == batch_size:
            yield seqs, labels
            seqs, labels = [], []

        seqs.append(sent_)
        labels.append(label_)

    # For last batch
    if len(seqs) != 0:
        yield seqs, labels


def pad_sequences(sequences, pad_mark=0):
    '''
    Sequence padding with zeros
    '''
    # The longest length in each batch
    max_length = max(map(lambda x: len(x), sequences))
    seq_list, seq_len_list = [], []
    for seq in sequences:
        seq = list(seq)
        seq_ = seq[: max_length] + [pad_mark] * max(max_length - len(seq), 0)
        seq_list.append(seq_)
        seq_len_list.append(min(len(seq), max_length))
    return seq_list, seq_len_list

