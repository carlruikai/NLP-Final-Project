import numpy as np
from lstm_crf_tagger import LSTM_CRF_Tagger


class LSTM_CRF_Tagger_ScienceExamCER(LSTM_CRF_Tagger):
    
    def __init__(self,
                 train_path,
                 model_path,
                 valid_path,
                 max_length,
                 embedding_dim,
                 epochs,
                 batch_size):
        super(LSTM_CRF_Tagger_ScienceExamCER, self).__init__(
            train_path=train_path,
            model_path=model_path,
            max_length=max_length,
            embedding_dim=embedding_dim,
            epochs=epochs,
            batch_size=batch_size
        )
        self.valid_path = valid_path
    
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
        for i, j in zip([0] + idx_list, idx_list + (
                [len(lines)] if idx_list[-1] != len(lines) else [])):
            sentence = []
            for line in lines[i: j - 1]:
                splitted = line[:-1].split(' ')
                words.add(splitted[0])
                pos_tags.add(splitted[2])
                ner_tags.add(splitted[5])
                sentence.append((splitted[0], splitted[1], splitted[2]))
            sentences.append(sentence)
        return sentences, words, pos_tags, ner_tags

    def train_valid_split(self, X_train, y_train):
        valid_sentences = self.load_file(self.valid_path)[0]
        X_valid, y_valid = self.preprocess(valid_sentences)
        return np.array(X_train), np.array(X_valid), \
            np.array(y_train), np.array(y_valid)
    

if __name__ == '__main__':

    LSTM_CRF_Tagger_ScienceExamCER(
        train_path='../../data/ScienceExamCER/train_spacy.txt',
        valid_path='../../data/ScienceExamCER/valid_spacy.txt',
        model_path='weights.best.science_exam_cer.hdf5',
        max_length=75,
        embedding_dim=20,
        epochs=5,
        batch_size=16,
        ).main(
        test_path='../../data/ScienceExamCER/train_spacy.txt'
        )
