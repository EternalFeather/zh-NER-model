import argparse


def str2bool(s):
    '''
    Transform str to bool type
    '''
    if s.lower() in ('yes', 'true', 'y', 't', 1):
        return True
    elif s.lower() in ('no', 'false', 'n', 'f', 0):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_entity(tag_seq, char_seq):
    '''
    Return: Key word tags
    '''
    IPT = get_IPT_entity(tag_seq, char_seq)
    return IPT


def get_IPT_entity(tag_seq, char_seq):
    '''
    Tags for IPT entity
    '''
    length = len(char_seq)
    IPT = []
    for i, (char, tag) in enumerate(zip(char_seq, tag_seq)):
        # flag for judging end signal
        if tag == 'B-IPT':
            if 'ipt' in locals().keys():
                IPT.append(ipt)
                del ipt
            ipt = char
            if i + 1 == length:
                IPT.append(ipt)
        if tag == 'I-IPT':
            ipt += char
            if i + 1 == length:
                IPT.append(ipt)
        # Only one word situation
        if tag not in ['I-IPT', 'B-PER']:
            if 'ipt' in locals().keys():
                IPT.append(ipt)
                del ipt
            continue
    return IPT



