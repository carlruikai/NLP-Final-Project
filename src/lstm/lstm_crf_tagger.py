import time
import numpy as np
from collections import defaultdict
from matplotlib import pyplot as plt

from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import Model, Input
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Bidirectional
from keras_contrib.layers import CRF
from keras.callbacks import ModelCheckpoint
from keras.utils import multi_gpu_model

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


def zero():
    return 0


def one():
    return 1


class LSTM_CRF_Tagger(object):

    def __init__(self,
                 train_path,
                 model_name,
                 max_length,
                 embedding_dim,
                 epochs,
                 batch_size,
                 n_gpu=None):
        
        self.model_name = model_name
        self.max_length = max_length
        self.embedding_dim = embedding_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.n_gpu = n_gpu
        
        self.sentences, self.words, self.pos_tags, self.ner_tags = \
            self.load_file(train_path)
        max_length = 0
        for s in self.sentences:
            max_length = max(max_length, len(s))
        print('Reset max_length: ', max_length)
        self.max_length = max_length
        
        self.n_words, self.n_tags = len(self.words), len(self.ner_tags)
        self.word_to_idx, self.idx_to_word, self.tag_to_idx, \
                self.idx_to_tag = self.get_dicts()

    def load_file(self, file_name):
        raise NotImplementedError
    
    def normalize_sentence(self, sentence):
        # sentence = sentence.lower()
        # sentence = sentence.split(' ')
        # if sentence[-1] == '\n':
        #     sentence.pop()
        return sentence
    
    def get_dicts(self):

        word_to_idx = defaultdict(
            one, {w: i + 2 for i, w in enumerate(self.words)})
        word_to_idx['UNK'], word_to_idx['PAD'] = 1, 0
        idx_to_word = defaultdict(
            one, {i: w for w, i in word_to_idx.items()})

        tag_to_idx = defaultdict(
            zero, {t: i+1 for i, t in enumerate(self.ner_tags)})
        tag_to_idx['PAD'] = 0
        idx_to_tag = defaultdict(
            zero, {i: w for w, i in tag_to_idx.items()})

        return word_to_idx, idx_to_word, tag_to_idx, idx_to_tag
    
    def preprocess(self, sentences):
                
        X = [[self.word_to_idx[w[0]] for w in s] for s in sentences]
        X = pad_sequences(maxlen=self.max_length,
                          sequences=X,
                          padding='post',
                          value=self.word_to_idx['PAD'])
        
        y = [[self.tag_to_idx[w[2]] for w in s] for s in sentences]
        y = pad_sequences(maxlen=self.max_length,
                          sequences=y,
                          padding='post',
                          value=self.tag_to_idx['PAD'])
        y = [to_categorical(y_, num_classes=self.n_tags + 1) for y_ in y]
        
        return np.array(X), np.array(y)
    
    def build_model(self):
        input = Input(shape=(self.max_length,))

        # Embedding Layer
        x = Embedding(input_dim=self.n_words + 2,
                      output_dim=self.embedding_dim,
                      input_length=self.max_length, 
                      mask_zero=True)(input)
        
        # Bi-directional LSTM Layer
        x = Bidirectional(LSTM(units=50,
                               return_sequences=True,
                               recurrent_dropout=0.1))(x)
        
        # Dense Layer
        x = TimeDistributed(Dense(50, activation='relu'))(x)
        
        # CRF Layer
        crf = CRF(self.n_tags + 1)
        out = crf(x)
        
        # Build model
        model = Model(input, out)
        
        # Multi-GPU
        # if self.n_gpu:
        #     model = multi_gpu_model(model, gpus=self.n_gpu)
        
        # Compile model
        model.compile(optimizer="rmsprop",
                      loss=crf.loss_function,
                      metrics=[crf.accuracy])
        
        # Show model summary
        model.summary()
        
        return model
    
    def train_valid_split(self, X_train, y_train):
        X_train, X_valid, y_train, y_valid = \
            train_test_split(X_train, y_train, test_size=0.1)
        return np.array(X_train), np.array(X_valid), \
            np.array(y_train), np.array(y_valid)
    
    def train(self):

        model = self.build_model()
        
        X_train, y_train = self.preprocess(self.sentences)
        
        X_train, X_valid, y_train, y_valid = \
            self.train_valid_split(X_train, y_train)
            
        print('Number of training data: ', len(y_train))
        print('Number of validation data: ', len(y_valid))
        
        model_path = 'weights.best.' + self.model_name + '.hdf5'
            
        checkpointer = ModelCheckpoint(
            filepath=model_path,
            verbose=1,
            save_best_only=True)
        
        model.fit(X_train, y_train,
                  validation_data=(X_valid, y_valid),
                  epochs=self.epochs,
                  batch_size=self.batch_size,
                  callbacks=[checkpointer],
                  verbose=1)
        
        history = model.history

        x_int = np.arange(1, len(history.history['crf_viterbi_accuracy']) + 1)
        
        plt.figure()
        plt.plot(x_int, history.history['crf_viterbi_accuracy'])
        plt.plot(x_int, history.history['val_crf_viterbi_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'valid'], loc='upper left')
        plt.savefig(self.model_name + '_accuracy.png')

        plt.figure()
        plt.plot(x_int, history.history['loss'])
        plt.plot(x_int, history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'valid'], loc='upper left')
        plt.savefig(self.model_name + '_loss.png')
        
        model.load_weights(model_path)

        return model

    def test(self, model, test_path, test_num):
        
        if test_num:
            test_sentences = self.load_file(test_path)[0][:test_num]
        else:
            test_sentences = self.load_file(test_path)[0]
            
        X_test, y_test = self.preprocess(test_sentences)
        y_preds = np.argmax(model.predict(X_test), axis=2)
        
        print('=' * 70)
        print('Testing...')
        accuracy = []
        y_true_all = []
        y_pred_all = []
        for y_true, sent, y_pred in zip(y_test, test_sentences, y_preds):
            sent_len = min(len(sent), len(y_true)) 
            y_true_short = y_true[:sent_len]
            y_pred_short = y_pred[:sent_len]
            y_true_all.extend(np.argmax(y_true_short, axis=1).tolist())
            y_pred_all.extend(y_pred_short.tolist())
            accuracy.extend((y_pred_short == np.argmax(y_true_short, axis=1)).tolist())
        
        # Compute the final accuracy
        accuracy = np.array(accuracy).mean()
        print('=' * 70)
        print('Accuracy: {:.4f}%'.format(accuracy * 100))
        print('=' * 70)
        print('Classification Report:')
        print('-' * 70)
        target_names = [self.idx_to_tag[i] for i in \
            sorted(list(set(y_true_all + y_pred_all)))]
        print(len(target_names))
        print(classification_report(
            y_true_all, y_pred_all, target_names=target_names))
        print('=' * 70)
    
    def reload_model_and_test(self, test_path, test_num=None):
        
        model_path = 'weights.best.' + self.model_name + '.hdf5'
        
        print('=' * 70)
        print('Rebuild model...')
        model = self.build_model()
        
        print('=' * 70)
        print('Load model weights...')
        model.load_weights(model_path)

        self.test(model, test_path, test_num)
        
    def main(self, test_path, test_num=None):
        start_time = time.time()
        
        # Train model
        print('=' * 70)
        print('Training...')
        model = self.train()
        
        self.test(model, test_path, test_num)

        print('Runtime: {:.2f}s'.format(time.time() - start_time))
        print('=' * 70)
