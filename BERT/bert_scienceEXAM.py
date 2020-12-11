import pandas as pd
from simpletransformers.ner.ner_model import NERModel, NERArgs
import joblib
import os
dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data/ScienceExamCER/")

#ScienceExamCER train dataset

def get_bert_train_sci():
  sci_train_words = []
  data = open(dir + 'train_spacy.txt', 'r', encoding="utf-8")
  lines = data.readlines()
  for line in lines:
    line = line.strip('\n').split('\n\t')
    for str in line:
      sub_str = str.split(' ')
    if sub_str:
      sci_train_words.append(sub_str)

  sci_train_refined_data = []
  sci_train_refined_data_unit = []
  sci_label = []
  i = 0
  for l in range(len(sci_train_words)):
    if sci_train_words[l][0] == '':
      i += 1
      continue
    sci_train_refined_data_unit.append(i)
    sci_train_refined_data_unit.append(sci_train_words[l][0])
    sci_train_refined_data_unit.append(sci_train_words[l][-1])
    sci_train_refined_data.append(sci_train_refined_data_unit)
    sci_label.append(sci_train_words[l][-1])
    sci_train_refined_data_unit = []

  sci_label = set(sci_label)
  sci_label = list(sci_label)

  sci_train_DataFrame = pd.DataFrame(data=sci_train_refined_data, columns=('sentence_id', 'words', 'labels'))
  return sci_train_DataFrame, sci_label


#ScienceExamCER validation dataset
def get_bert_val_sci():

  sci_val_words = []
  data = open(dir + 'valid_spacy.txt', 'r', encoding="utf-8")
  lines = data.readlines()
  for line in lines:
    line = line.strip('\n').split('\n\t')
    for str in line:
      sub_str = str.split(' ')
    if sub_str:
      sci_val_words.append(sub_str)


  sci_val_refined_data = []
  sci_val_refined_data_unit = []
  i = 0
  for l in range(len(sci_val_words)):
    if sci_val_words[l][0] == '':
      i += 1
      continue
    sci_val_refined_data_unit.append(i)
    sci_val_refined_data_unit.append(sci_val_words[l][0])
    sci_val_refined_data_unit.append(sci_val_words[l][-1])
    sci_val_refined_data.append(sci_val_refined_data_unit)
    sci_val_refined_data_unit = []

  sci_val_DataFrame = pd.DataFrame(data=sci_val_refined_data, columns=('sentence_id', 'words', 'labels'))
  return sci_val_DataFrame


#ScienceExamCER test dataset
def get_bert_test_sci(sci_label):

  sci_test_words = []
  data = open('test_spacy.txt', 'r', encoding="utf-8")
  lines = data.readlines()
  for line in lines:
    line = line.strip('\n').split('\n\t')
    for str in line:
      sub_str = str.split(' ')
    if sub_str:
      sci_test_words.append(sub_str)


  sci_test_refined_data = []
  sci_test_refined_data_unit = []
  i = 0
  for l in range(len(sci_test_words)):
    if sci_test_words[l][0] == '':
      i += 1
      continue
    sci_test_refined_data_unit.append(i)
    sci_test_refined_data_unit.append(sci_test_words[l][0])
    if sci_test_words[l][-1] not in sci_label:
      sci_test_words[l][-1] = 'O'
    sci_test_refined_data_unit.append(sci_test_words[l][-1])
    sci_test_refined_data.append(sci_test_refined_data_unit)
    sci_test_refined_data_unit = []

  sci_test_DataFrame = pd.DataFrame(data=sci_test_refined_data, columns=('sentence_id', 'words', 'labels'))
  return sci_test_DataFrame


if __name__ == '__main__':
  sci_train_DataFrame, sci_label = get_bert_train_sci()
  sci_val_DataFrame = get_bert_val_sci()
  sci_test_DataFrame = get_bert_test_sci(sci_label)

  # Create a NERModel
  sci_model_args = NERArgs()
  sci_model_args.labels_list = sci_label
  sci_model_args.overwrite_output_dir = True
  sci_model = NERModel('bert', 'bert-base-cased', args=sci_model_args)

  # Train the model
  sci_model.train_model(sci_train_DataFrame, eval_data=sci_val_DataFrame)

  # Evaluate the model
  sci_result, sci_model_outputs, sci_predictions = sci_model.eval_model(sci_test_DataFrame)

  # Check predictions
  print(sci_result)
  print(sci_predictions[:5])

  joblib.dump(sci_model, "ScienceExam_bert.joblib")

