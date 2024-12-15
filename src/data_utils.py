import random
import re
import string

from src.stop_words import ukr_stop_words


def clean_string(input_string):
    allowed_characters = 'а-щА-ЩЬьЮюЯяЇїІіЄєҐґ' + string.punctuation

    input_string = input_string.lower()

    if input_string.startswith('#'):
        return ''

    if "ы" in input_string:
        return ''

    if not re.search(r'[а-щА-ЩЬьЮюЯяЇїІіЄєҐґ]', input_string):
        return ''

    input_string = re.sub(rf'[^{allowed_characters} ]', '', input_string)
    input_string = re.sub(r'[0-9a-z,*_$%&()/@^+.]', '', input_string)
    input_string = re.sub(r'[\xad]', '', input_string)
    input_string = input_string.lstrip(string.punctuation + '–' + '−')
    input_string = input_string.rstrip(string.punctuation + '–' + '−')
    input_string = re.sub(r'(.)\1{3,}', r'\1\1\1', input_string)
    input_string = re.sub(r'((.)-\2-)\2(-\2)+', r'\1\3', input_string)
    input_string = re.sub(r'-{2,}', '-', input_string)

    if len(input_string) < 3:
        return ''

    if input_string in ukr_stop_words:
        return ''

    return input_string


def read_corpus(path, clean=True, MR=True, encoding='utf8', shuffle=False, lower=True):
    data = []
    labels = []
    with open(path, encoding=encoding) as fin:
        for line in fin:
            if MR:
                label, _, text = line.partition(' ')
                label = int(label)
            else:
                label, _, text = line.partition(',')
                label = int(label) - 1
            if clean:
                text = clean_string(text.strip()) if clean else text.strip()
            if lower:
                text = text.lower()
            labels.append(label)
            data.append(text.split())

    if shuffle:
        perm = list(range(len(data)))
        random.shuffle(perm)
        data = [data[i] for i in perm]
        labels = [labels[i] for i in perm]

    return data, labels
