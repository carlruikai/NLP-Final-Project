import numpy as np
from lstm_crf_tagger import LSTM_CRF_Tagger


class LSTM_CRF_Tagger_ScienceExamCER(LSTM_CRF_Tagger):
    
    def __init__(self,
                 train_path,
                 model_name,
                 valid_path,
                 max_length,
                 embedding_dim,
                 epochs,
                 batch_size,
                 n_gpu=None):
        super(LSTM_CRF_Tagger_ScienceExamCER, self).__init__(
            train_path=train_path,
            model_name=model_name,
            max_length=max_length,
            embedding_dim=embedding_dim,
            epochs=epochs,
            batch_size=batch_size,
            n_gpu=None
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
                sentence.append((splitted[0], splitted[2], splitted[5]))
            sentences.append(sentence)
        return sentences, words, pos_tags, ner_tags

    def train_valid_split(self, X_train, y_train):
        valid_sentences = self.load_file(self.valid_path)[0]
        X_valid, y_valid = self.preprocess(valid_sentences)
        
        return np.array(X_train), np.array(X_valid), \
            np.array(y_train), np.array(y_valid)
    

if __name__ == '__main__':
    
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    LSTM_CRF_Tagger_ScienceExamCER(
        train_path='../../data/ScienceExamCER/train_spacy.txt',
        valid_path='../../data/ScienceExamCER/valid_spacy.txt',
        model_name='science_exam_cer',
        max_length=100,
        embedding_dim=20,
        epochs=50,
        batch_size=64,
        n_gpu=None
        ).main(
        test_path='../../data/ScienceExamCER/train_spacy.txt'
        )
