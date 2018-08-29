import argparse
from string import punctuation as p


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
    NOR = get_NOR_entity(tag_seq, char_seq)
    VER = get_VER_entity(tag_seq, char_seq)
    ENG = get_ENG_entity(tag_seq, char_seq)
    OTH = get_OTH_entity(tag_seq, char_seq)
    return NOR, VER, ENG, OTH


def get_NOR_entity(tag_seq, char_seq):
    '''
    Tags for NOR entity
    '''
    NOR = []
    stop_p = p + "~·！@#￥%……&*（）——=+-{}【】：；“”‘’《》，。？、|、"
    char_seq = [char for char in char_seq if char not in stop_p]
    length = len(char_seq)
    for i, (char, tag) in enumerate(zip(char_seq, tag_seq)):
        # flag for judging end signal
        if tag == 'B-NOR':
            if 'nor' in locals().keys():
                NOR.append(nor)
                del nor
            nor = char
            if i + 1 == length:
                NOR.append(nor)
                del nor
        elif tag == 'I-NOR':
            if 'nor' not in locals().keys():
                continue
            nor += char
            if i + 1 == length:
                NOR.append(nor)
                del nor
        elif tag not in ['I-NOR', 'B-NOR']:
            if 'nor' in locals().keys():
                NOR.append(nor)
                del nor
        else:
            print('Judgement Exception ... (Please debug)')
    return set(NOR)


def get_VER_entity(tag_seq, char_seq):
    '''
    Tags for VER entity
    '''
    VER = []
    stop_p = p + "~·！@#￥%……&*（）——=+-{}【】：；“”‘’《》，。？、|、"
    char_seq = [char for char in char_seq if char not in stop_p]
    length = len(char_seq)
    for i, (char, tag) in enumerate(zip(char_seq, tag_seq)):
        # flag for judging end signal
        if tag == 'B-VER':
            if 'ver' in locals().keys():
                VER.append(ver)
                del ver
            ver = char
            if i + 1 == length:
                VER.append(ver)
                del ver
        elif tag == 'I-VER':
            if 'ver' not in locals().keys():
                continue
            ver += char
            if i + 1 == length:
                VER.append(ver)
                del ver
        elif tag not in ['I-VER', 'B-VER']:
            if 'ver' in locals().keys():
                VER.append(ver)
                del ver
        else:
            print('Judgement Exception ... (Please debug)')
    return set(VER)


def get_ENG_entity(tag_seq, char_seq):
    '''
    Tags for ENG entity
    '''
    ENG = []
    stop_p = p + "~·！@#￥%……&*（）——=+-{}【】：；“”‘’《》，。？、|、"
    char_seq = [char for char in char_seq if char not in stop_p]
    length = len(char_seq)
    for i, (char, tag) in enumerate(zip(char_seq, tag_seq)):
        # flag for judging end signal
        if tag == 'B-ENG':
            if 'ipt' in locals().keys():
                ENG.append(eng)
                del eng
                eng = char
            if i + 1 == length:
                ENG.append(eng)
                del eng
        elif tag == 'I-ENG':
            if 'eng' not in locals().keys():
                continue
            eng += char
            if i + 1 == length:
                ENG.append(eng)
                del eng
        elif tag not in ['I-ENG', 'B-ENG']:
            if 'eng' in locals().keys():
                ENG.append(eng)
                del eng
        else:
            print('Judgement Exception ... (Please debug)')
    return set(ENG)


def get_OTH_entity(tag_seq, char_seq):
    '''
    Tags for OTH entity
    '''
    OTH = []
    stop_p = p + "~·！@#￥%……&*（）——=+-{}【】：；“”‘’《》，。？、|、"
    char_seq = [char for char in char_seq if char not in stop_p]
    length = len(char_seq)
    for i, (char, tag) in enumerate(zip(char_seq, tag_seq)):
        # flag for judging end signal
        if tag == 'B-OTH':
            if 'oth' in locals().keys():
                OTH.append(oth)
                del oth
            oth = char
            if i + 1 == length:
                OTH.append(oth)
                del oth
        elif tag == 'I-OTH':
            if 'oth' not in locals().keys():
                continue
                oth += char
            if i + 1 == length:
                OTH.append(oth)
                del oth
        elif tag not in ['I-OTH', 'B-OTH']:
            if 'oth' in locals().keys():
                OTH.append(oth)
                del oth
        else:
            print('Judgement Exception ... (Please debug)')
    return set(OTH)