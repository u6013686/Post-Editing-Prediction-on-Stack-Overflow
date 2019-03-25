from control_group_model.generate_dataset import create_dataset_LSTM,create_dataset_fasttext
from control_group_model import attentionbased_LSTM, linear_model,fasttext

pattern = 'content'
create_dataset_LSTM(pattern)
create_dataset_fasttext(pattern)
attentionbased_LSTM.train_model(pattern)
linear_model.train_model(pattern, 'hinge') # SVM
linear_model.train_model(pattern, 'log') # Logistic Regression
#fasttext.train_model(pattern)
