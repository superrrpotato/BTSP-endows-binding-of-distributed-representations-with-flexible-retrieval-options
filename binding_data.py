

import torch
import numpy as np    

def generate_words_based_dataset(num_sentences:int, Nw: int, L: int, K: int, fp: float, device: None):
    """
    Generates a dataset of words based on the given parameters.

    Args:
        num_sentences (int): The number of sentences to generate.
        Nw (int): The number of words in the vocabulary.
        L (int): The length of each word.
        K (int): The number of words in each sentence.
        fp (float): The fraction of ones in each word.
        device (None): The device to store the generated data.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing the generated data and the vocabulary, both stored on the specified device.
    """

    num_ones_per_row = int(L * fp)
    # print(f"num_ones_per_row: {num_ones_per_row}")
    voc = torch.zeros(Nw, L).cuda().float()

    for i in range(Nw):
        row_indices = torch.randperm(L)[:num_ones_per_row]
        voc[i, row_indices] = 1.
    indice_list = range(Nw)
    data = []

    #### words randomly combined ####
    # for i in range(num_sentences):
    #     selected_words = np.random.choice(indice_list, K, replace=False)
    #     sentence = []
    #     for word_idx in selected_words:
    #         sentence.append(voc[word_idx])
    #     sentence = torch.cat(sentence, dim=0)
    #     data.append(sentence)

    # #### sentences with unique prefix ####
    id = torch.randperm(Nw*Nw)[:num_sentences]
    for i in range(num_sentences):
        selected_words = np.random.choice(indice_list, K-2, replace=False)
        sentence = []
        flat_idx = id[i]
        x_idx = flat_idx % Nw
        y_idx = flat_idx // Nw
        x = voc[x_idx]
        y = voc[y_idx]
        sentence = [x, y]
        for word_idx in selected_words:
            sentence.append(voc[word_idx])
        sentence = torch.cat(sentence, dim=0)
        data.append(sentence)


    data = torch.stack(data,dim=0)
    return data.to(device), voc.to(device)


def masked_input_w_two_cues(data, K):
    """
    Generates masked input by randomly selecting two cues in each sentence.

    Args:
        data (torch.Tensor): The input data.
        K (int): The number of words in each sentence.

    Returns:
        torch.Tensor: The masked input with two cues in each sentence.
    """

    num_sentences = data.shape[0]
    masked_input = []
    for i in range(num_sentences):
        masked_sentence = []
        words = torch.chunk(data[i], K, dim=0)
        # cue_idx = torch.randperm(K)[:2]
        cue_idx = [0, 1] # select first two words as the unique cue-pair (w/ sentences generated with unique prefix)
        for j in range(K):
            if j in cue_idx:
                masked_sentence.append(words[j])
            else:
                masked_sentence.append(torch.zeros_like(words[j]))
        masked_sentence = torch.cat(masked_sentence, dim=0)
        masked_input.append(masked_sentence)
    masked_input = torch.stack(masked_input, dim=0)
    return masked_input