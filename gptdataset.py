import torch
import random
import numpy as np
import h5py
import os
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, Dataset
import tokenization


def create_instances_from_document(documents, max_seq_length, short_seq_prob, rng):
    max_num_tokens = max_seq_length - 3 # Account for [CLS], [SEP], [SEP]
    target_seq_length = max_num_tokens
    if rng.random() < short_seq_prob:
        target_seq_length = rng.randint(2, max_num_tokens)
    instances = []
    current_chunk = []
    current_length = 0
    i = 0
    while i < len(documents):
        segment = documents[i]
        current_chunk.append(segment)
        current_length += len(segment)
        if i == len(documents) - 1 or current_length >= target_seq_length:
            if current_chunk:
                a_end = 1
                if len(current_chunk) >= 2:
                    a_end = rng.randint(1, len(current_chunk) - 1)
                tokens_a = []
                for j in range(a_end):
                    tokens_a.extend(current_chunk[j])
                assert len(tokens_a) >= 1
                tokens = []
                tokens.append("[CLS]")
                for token in tokens_a:
                    tokens.append(token)
                    tokens.append("[SEP]")
                tokens.append("[SEP]")
                instances.append(tokens)
            current_chunk = []
            current_length = 0
        i += 1
    return instances


def create_training_instances(input_files, tokenizer, max_seq_length,
        short_seq_prob, rng):
    all_documents = [[]]
    for input_file in input_files:
        print('processing ' + str(input_file))
        with open(input_file, 'r') as f:
            while True:
                line = tokenization.convert_to_unicode(f.readline())
                if not line:
                    break
                line = line.strip()
                if not line:  # Empty lines are used as document delimiters
                    all_documents.append([])
                tokens = tokenizer.tokenize(line)
                if tokens:
                    all_documents[-1].append(tokens)
    all_documents = [x for x in all_documents if x]
    all_documents_ = []
    for item in all_documents:
        all_documents_.extend(item)
    all_documents = all_documents_
    # vocab_words = list(tokenizer.vocab.keys())
    instances = create_instances_from_document(all_documents, max_seq_length, short_seq_prob, rng)
    rng.shuffle(instances)
    print(len(instances))
    print(instances[123])
    return instances


# class GPTDataset(Dataset):
#     def __init__(self, input_file, max_pred_length):
#         self.input_file = input_file
#         self.max_pred_length = max_pred_length
#         f = h5py.File(input_file, 'r')
#         keys = [
#             'input_ids',
#             'input_mask',
#             'segment_ids',
#             'masked_lm_positions',
#             'masked_lm_ids',
#             'next_sentence_labels',
#         ]
#         self.inputs = [np.asarray(f[key][:]) for key in keys]
#         print(f"Loaded {len(self.inputs[0]):d} samples from file: {input_file}")
#         f.close()

#     def __len__(self):
#         return len(self.inputs[0])
    
#     def __getitem__(self, index):
#         [
#             input_ids,
#             input_mask,
#             segment_ids,
#             masked_lm_positions,
#             masked_lm_ids,
#             next_sentence_labels,
#         ] = [
#             torch.from_numpy(input[index].astype(np.int64))
#             if indice < 5 else torch.from_numpy(np.asarray(input[index].astype(np.int64)))
#             for indice, input in enumerate(self.inputs)
#         ]
#         masked_lm_labels = torch.zeros(input_ids.shape, dtype=torch.long) - 100
#         index = self.max_pred_length
#         masked_token_count = torch.count_nonzero(masked_lm_positions)
#         if masked_token_count != 0:
#             index = masked_token_count
#         masked_lm_labels[masked_lm_positions[:index]] = masked_lm_ids[:index]
#         return [
#             input_ids,
#             segment_ids,
#             input_mask,
#             masked_lm_labels,
#             next_sentence_labels,
#         ]


# def get_pretraining_datafiles(dirname):
#     datafiles = [os.path.join(dirname, f) for f in os.listdir(dirname) \
#         if os.path.isfile(os.path.join(dirname, f)) and 'pretrain-part' in f]
#     datafiles.sort()
#     return datafiles


# def get_pretraining_dataloader(input_file, batch_size, max_pred_length):
#     dataset = PretrainingDataset(input_file, max_pred_length)
#     sampler = RandomSampler(dataset)
#     loader = DataLoader(dataset, sampler=sampler, batch_size=batch_size)
#     return loader


if __name__ == '__main__':
    tokenizer = tokenization.FullTokenizer('./datasets/multilingual-cased-vocab.txt')
    rng = random.Random(0)
    create_training_instances(
        ['./datasets/authorliu/results/pretrain-part-00000.txt'],
        tokenizer,
        512,
        0.1,
        rng)

    # datafiles = get_pretraining_datafiles('/data/wiki/results/hdf5')
    # dataloader = get_pretraining_dataloader(datafiles[0], 10, 32)
    # for data in dataloader:
    #     [
    #         input_ids,
    #         segment_ids,
    #         input_mask,
    #         masked_lm_labels,
    #         next_sentence_labels,
    #     ] = data
    #     idss = input_ids.tolist()
    #     for ids in idss:
    #         tokens = tokenizer.convert_ids_to_tokens(ids)
    #         print(" ".join(tokens))
    #     # print(data[0])
    #     raise
