from lstm_crf_tagger import LSTM_CRF_Tagger

class LSTM_CRF_Tagger_CoNLL2000(LSTM_CRF_Tagger):
    
    def __init__(self,
                 train_path,
                 max_length,
                 embedding_dim,
                 epochs,
                 batch_size):
        super(LSTM_CRF_Tagger_CoNLL2000, self).__init__(
            train_path=train_path,
            max_length=max_length,
            embedding_dim=embedding_dim,
            epochs=epochs,
            batch_size=batch_size
        )
    
    def load_file(self, file_name):
        print('=' * 70)
        print('Loading files...')
        with open(file_name, "r") as f:
            lines = f.readlines()
        idx_list = [i + 1 for i, v in enumerate(lines) if v == '\n']
        sentences = []
        words = set()
        pos_tags = set()
        ner_tags = set()
        max_length = 0
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
            max_length = max(max_length, len(sentence))
        print('Reset max_length: ', max_length)
        self.max_length = max_length
        return sentences, words, pos_tags, ner_tags


if __name__ == '__main__':

    LSTM_CRF_Tagger_CoNLL2000(
        train_path='../../data/CoNLL-2000/train.txt',
        max_length=75,
        embedding_dim=20,
        epochs=50,
        batch_size=256
        ).main(
        test_path='../../data/CoNLL-2000/test.txt'
        )
