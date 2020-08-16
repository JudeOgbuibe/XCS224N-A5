#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch.nn as nn


# Do not change these imports; your module names should be
#   `CNN` in the file `cnn.py`
#   `Highway` in the file `highway.py`
# Uncomment the following two imports once you're ready to run part 1(f)

from cnn import CNN
from highway import Highway

# End "do not change" 

class ModelEmbeddings(nn.Module):
    """
    Class that converts input words to their CNN-based embeddings.
    """

    def __init__(self, embed_size, vocab):
        """
        Init the Embedding layer for one language
        @param embed_size (int): Embedding size (dimensionality) for the output 
        @param vocab (VocabEntry): VocabEntry object. See vocab.py for documentation.
        """
        super(ModelEmbeddings, self).__init__()

        ## A4 code
        # pad_token_idx = vocab.src['<pad>']
        # self.embeddings = nn.Embedding(len(vocab.src), embed_size, padding_idx=pad_token_idx)
        ## End A4 code

        ### YOUR CODE HERE for part 1f
        self.embed_size = embed_size
        self.char_emb_size = 50
        self.kernel_size = 5
        self.dropout = 0.3
        self.max_word_length = 21
        # self.filter_size = 
        pad_token_idx = vocab.char2id['<pad>']
        input_size = len(vocab.char2id)
        self.embeddings = nn.Embedding(input_size, self.char_emb_size, padding_idx=pad_token_idx)
        self.cnn = CNN(self.kernel_size, self.embed_size, self.char_emb_size)
        self.highway = Highway(self.embed_size)
        self.dropout = nn.Dropout(0.3)


        ### END YOUR CODE

    def forward(self, input_tensor):
        """
        Looks up character-based CNN embeddings for the words in a batch of sentences.
        @param input_tensor: Tensor of integers of shape (sentence_length, batch_size, max_word_length) where
            each integer is an index into the character vocabulary

        @param output: Tensor of shape (sentence_length, batch_size, embed_size), containing the 
            CNN-based embeddings for each word of the sentences in the batch
        """
        ## A4 code
        # output = self.embeddings(input)
        # return output
        ## End A4 code

        ### YOUR CODE HERE for part 1f
        #print(input_tensor.size())
        input_embed = self.embeddings(input_tensor)
        #print(input_embed.size())
        sentence_length, batch_size, mwl, ces = input_embed.size()
        #input_cnn = input_embed.permute(3,2, 0,1)
        input_cnn = input_embed.view(-1, ces, mwl)
        #print(input_cnn.size())
        input_highway = self.cnn(input_cnn)
        input_dropout = self.highway(input_highway)
        output = self.dropout(input_dropout)

        return output.view(sentence_length, batch_size, self.embed_size)
        ### END YOUR CODE
