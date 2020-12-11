import pandas as pd
from simpletransformers.ner.ner_model import NERModel, NERArgs
import joblib
import os
dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data/CoNLL-2000/")

# conll train dataset

def get_bert_train_conll():
  words = []
  data = open('train.txt', 'r')
  lines = data.readlines()
  for line in lines:
    line = line.strip('\n').split('\n\t')
    for str in line:
      sub_str = str.split(' ')
    if sub_str:
      words.append(sub_str)


  refined_data = []
  refined_data_unit = []
  label = []
  i = 0
  for l in range(len(words)):
    if words[l][0] == '':
      continue
    if words[l][0] == ',':
      pass
    if words[l][0] == '.':
      i += 1
      continue
    refined_data_unit.append(i)
    refined_data_unit.append(words[l][0])
    refined_data_unit.append(words[l][2])
    refined_data.append(refined_data_unit)
    label.append(words[l][2])
    refined_data_unit = []

  label = set(label)
  label = list(label)

  DataFrame = pd.DataFrame(data=refined_data, columns=('sentence_id', 'words', 'labels'))
  return DataFrame, label

# conll get test dataset
def get_bert_test_conll():
  test_words = []
  test_data = open('test.txt', 'r')
  test_lines = test_data.readlines()
  for line in test_lines:
    line = line.strip('\n').split('\n\t')
    for str in line:
      sub_str = str.split(' ')
    if sub_str:
      test_words.append(sub_str)


  test_refined_data = []
  test_refined_data_unit = []
  test_label = []
  i = 0
  for l in range(len(test_words)):
    if test_words[l][0] == '':
      continue
    if test_words[l][0] == ',':
      pass
    if test_words[l][0] == '.':
      i += 1
      continue
    test_refined_data_unit.append(i)
    test_refined_data_unit.append(test_words[l][0])
    test_refined_data_unit.append(test_words[l][2])
    test_refined_data.append(test_refined_data_unit)
    test_label.append(test_words[l][2])
    test_refined_data_unit = []


  test_DataFrame = pd.DataFrame(data=test_refined_data, columns=('sentence_id', 'words', 'labels'))

  return test_DataFrame

if __name__ == '__main__':
    
  train_DataFrame, label = get_bert_train_conll()

  test_DataFrame = get_bert_test_conll()

  # create model
  model_args = NERArgs()
  model_args.labels_list = label
  model = NERModel('bert', 'bert-base-cased', args=model_args)

  # Train the model
  model.train_model(train_DataFrame)
  # Evaluate the model
  result, model_outputs, predictions = model.eval_model(test_DataFrame)
  joblib.dump(model, "conll2000_bert.joblib")