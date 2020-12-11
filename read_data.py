import os


def load_file_conll(file_name):
    dir = os.path.dirname(os.path.abspath(__file__)) + "/" + "data/CoNLL-2000/"
    file_name = dir + file_name
    with open(file_name, "r") as f:
        lines = f.readlines()
    idx_list = [i + 1 for i, v in enumerate(lines) if v == '\n']
    sentences = []
    words = set()
    pos_tags = set()
    ner_tags = set()
    for i, j in zip([0] + idx_list, idx_list + (
            [len(lines)] if idx_list[-1] != len(lines) else [])):
        sentence = []
        for line in lines[i: j - 1]:
            splitted = line[:-1].split(' ')
            words.add(splitted[0])
            pos_tags.add(splitted[1])
            ner_tags.add(splitted[2])
            sentence.append((splitted[0], splitted[1], splitted[2]))
        sentences.append(sentence)
    return sentences, words, pos_tags, ner_tags


def load_file_science(file_name):
    dir = os.path.dirname(os.path.abspath(__file__)) + "/" +"data/ScienceExamCER/"
    file_name = dir + file_name

    with open(file_name, "r", encoding="utf-8") as f:
        lines = f.readlines()
    idx_list = [i + 1 for i, v in enumerate(lines) if v == '\n']
    sentences = []
    words = set()
    pos_tags = set()
    ner_tags = set()
    for i, j in zip([0] + idx_list, idx_list + (
            [len(lines)] if idx_list[-1] != len(lines) else [])):
        sentence = []
        for line in lines[i: j - 1]:
            splitted = line[:-1].split(' ')
            words.add(splitted[0])
            pos_tags.add(splitted[2])
            ner_tags.add(splitted[5])
            sentence.append((splitted[0], splitted[2], splitted[5]))
        sentences.append(sentence)
    return sentences, words, pos_tags, ner_tags