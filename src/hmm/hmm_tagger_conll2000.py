import argparse
from hmm_tagger import HMMTagger

class HMMTaggerCoNLL2000(HMMTagger):
    
    def __init__(self,
                 train_path,
                 intermediate_path,
                 load_vars=False,
                 beam_search_n=None):
        super(HMMTaggerCoNLL2000, self).__init__(
            train_path=train_path,
            intermediate_path=intermediate_path,
            load_vars=load_vars,
            beam_search_n = beam_search_n
        )
    
    def load_file(self, file_name):
        print('=' * 70)
        print('Loading files...')
        with open(file_name, "r") as f:
            lines = f.readlines()
        idx_list = [i + 1 for i, v in enumerate(lines) if v == '\n']
        sentences = []
        for i, j in zip([0] + idx_list, idx_list + (
                [len(lines)] if idx_list[-1] != len(lines) else [])):
            sentence = []
            for line in lines[i: j - 1]:
                words = line[:-1].split(' ')
                sentence.append((words[0], words[2]))
            sentences.append(sentence)
        return sentences


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--beam_size', type=int, metavar='')
    args = parser.parse_args()

    HMMTaggerCoNLL2000(
        train_path='../../data/CoNLL-2000/train.txt',
        intermediate_path='../../data/variables/',
        load_vars=False,
        beam_search_n=args.beam_size
        ).main(
        test_path='../../data/CoNLL-2000/test.txt'
        )
