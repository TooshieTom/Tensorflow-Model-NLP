#Eli Chesnut, Tom Kerson, Ani Valluru

# This program is where we did training and testing of the models. The code can be all over the place at times due to changes for each trained model

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
from transformers import TFBertModel, BertTokenizer
import torch
import numpy as np
from itertools import combinations
import random
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from keras.regularizers import l2
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

folder_path = ""
df = pd.read_csv(folder_path)

model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
bert_model = TFBertModel.from_pretrained(model_name)

# function gets the vector for a word in a sentence
def get_word_vec(word, sentence, tokenizer):
    inputs = tokenizer(sentence, return_tensors="pt")
    with torch.no_grad():
        outputs = bert_model(**inputs)

    hidden_states = outputs.last_hidden_state.squeeze(0)
    
    # Find the index of the target word in the tokenized input
    word_idx = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0]).index(word)
    
    # Get the vector for the target word
    word_vector = hidden_states[word_idx][:50]#can add [:50] to limit the size of the vector
    return word_vector

# ---------------- Training of models done here -------------------------

def get_df_of_word():
    sentences = df['Tweet'].to_numpy().tolist()
    
    # Tokenize the sentences (convert them to BERT's input format)
    inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors="tf", max_length=512)

    # Pass tokenized inputs through the BERT model
    outputs = bert_model(**inputs)

    # Extract CLS token embeddings (from the last hidden state)
    cls_embeddings = outputs.last_hidden_state[:, 0, :]  # CLS token embeddings (batch_size, hidden_size)

    # Now `cls_embeddings` contains the embeddings for each sentence
    # print(cls_embeddings)
    # print(word_vectors[0])
    
    x = pd.DataFrame(cls_embeddings)

    return x
    
    pass


def read_file(word): # reads in files given to us and returns predictions and accuracy given the specific model

    # Get the sentences from the txt file
    with open(word+'_test.txt', 'r', encoding='utf-8') as file:
        # Read all lines from the file and strip any leading/trailing whitespace
        sentences = [line.strip() for line in file.readlines()]

    return sentences

def get_embeds(sentences):
    # Tokenize the sentences (convert them to BERT's input format)
    inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors="tf", max_length=512)

    # Pass tokenized inputs through the BERT model
    outputs = bert_model(**inputs)

    # Extract CLS token embeddings (from the last hidden state)
    cls_embeddings = outputs.last_hidden_state[:, 0, :]  # CLS token embeddings (batch_size, hidden_size)

    x = pd.DataFrame(cls_embeddings)
    
    x = np.array(x, dtype=np.float32)

    return x

X_overtime = get_df_of_word()
y_overtime = df['Party'].replace({'Democrat': '0', 'Republican': '1'})


# # Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_overtime, y_overtime, test_size=0.2, stratify=y_overtime, random_state=42)


# Instantiate the model
model = Sequential()
model.add(Dense(128, input_dim=768, activation='relu'))  # input_dim=768 for BERT embeddings
model.add(Dropout(0.2)) # add some dropout for regularization after conv layers
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))  # For binary classification, change if needed

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
                  loss='binary_crossentropy', 
                  metrics=['accuracy'])

print(X_train.shape)  # Ensure it has shape (num_samples, embedding_size)
X_train = np.array(X_train, dtype=np.float32)  # Ensure it's float32
y_train = np.array(y_train, dtype=np.float32)
# Also do this for y_train if it's necessary
X_test = np.array(X_test, dtype=np.float32)
y_test = np.array(y_test, dtype=np.float32)

# Fit the model on the data
model.fit(X_train, y_train, epochs=5, batch_size=2, verbose=1, validation_data=(X_test,y_test) )

# model.save('D:/College/Computer Science/NLP/Program 3/overtime_model2.h5')  # .h5 format is common for saving a model

# ------ Functions for each word --------------


# Load the model from the .h5 file
# model = tf.keras.models.load_model('overtime_model.h5')

# inputs = read_file("overtime")

# # Get model predictions (probabilities for class 1)
# predictions = model.predict(inputs)

# # Round predictions to get binary output (0 or 1)
# binary_predictions = np.round(predictions).astype(int)

# binary_predictions = binary_predictions + 1

# # Print the predictions
# print(binary_predictions)