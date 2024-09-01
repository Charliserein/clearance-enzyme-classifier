import sys
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from Bio import SeqIO

# Define the list of valid amino acids
AMINO_ACIDS = list("ARNDCEQGHILKMFPSTWYV")

def load_sequences(fasta_file, amino_acids):


    sequences = []
    for record in SeqIO.parse(fasta_file, 'fasta'):
        sequence = " ".join([item for item in record.seq if item in amino_acids])
        sequences.append(sequence)
    return sequences

def tokenize_sequences(sequences, max_nb_chars):

    tokenizer = Tokenizer(num_words=max_nb_chars)
    tokenizer.fit_on_texts(sequences)
    tokenized_sequences = tokenizer.texts_to_sequences(sequences)
    return tokenizer, tokenized_sequences

def pad_and_convert_to_tensor(sequences, max_sequence_length):

    padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length, padding='post')
    return torch.from_numpy(np.array(padded_sequences))

def main(fasta_file, output_file):
    sequences = load_sequences(fasta_file, AMINO_ACIDS)
    
    tokenizer, tokenized_sequences = tokenize_sequences(sequences, max_nb_chars=21)
    

    print(f'Found {len(tokenizer.word_index)} unique tokens.')
    print(tokenizer.word_index)
    
    sequence_tensor = pad_and_convert_to_tensor(tokenized_sequences, max_sequence_length=6269)
    
    torch.save(sequence_tensor, output_file)
    print(f'Saved tensor of shape {sequence_tensor.shape} to {output_file}')

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py <input_fasta_file> <output_tensor_file>")
        sys.exit(1)
    
    fasta_file = sys.argv[1]
    output_file = sys.argv[2]
    
    main(fasta_file, output_file)
