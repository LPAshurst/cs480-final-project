# import torch


# class CBOWDataset(torch.utils.data.Dataset):
#     def __init__(
#         self, tokenized_dataset, batch_size, context_size=2, pad_token_id=50256
#     ):
#         self.context = []
#         self.center = []
#         temp_context = []
#         temp_center = []

#         self.context_size = context_size
#         self.batch_size = batch_size
#         for input_ids in tokenized_dataset["input_ids"]:
#             tokens = [tok for tok in input_ids if tok != pad_token_id]
#             for i in range(context_size, len(tokens) - context_size):
#                 if len(temp_context) == batch_size:
#                     self.context.append(torch.stack(temp_context, dim=0))
#                     self.center.append(torch.stack(temp_center, dim=0))
#                     temp_context = []
#                     temp_center = []
#                 left = tokens[i - context_size : i]
#                 right = tokens[i + 1 : i + context_size + 1]
#                 context_tokens = left + right
#                 temp_context.append(torch.tensor(context_tokens, dtype=torch.long))
#                 center_token = tokens[i]
#                 temp_center.append(torch.tensor(center_token, dtype=torch.long))

#     def __len__(self):
#         return len(self.context)

#     def __getitem__(self, idx):
#         return {
#             "contexts": self.context[idx],
#             "centers": self.center[idx],
#         }


import torch


class CBOWDataset(torch.utils.data.Dataset):
    def __init__(self, tokenized_dataset, context_size=2, pad_token_id=50256):
        self.examples = []
        self.context_size = context_size

        for input_ids in tokenized_dataset["input_ids"]:
            tokens = [tok for tok in input_ids if tok != pad_token_id]
            for i in range(context_size, len(tokens) - context_size):
                left = tokens[i - context_size : i]
                right = tokens[i + 1 : i + context_size + 1]
                context_tokens = left + right  # List of 2*context_size tokens
                center_token = tokens[i]  # Single int token
                self.examples.append((context_tokens, center_token))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        context_tokens, center_token = self.examples[idx]
        return {
            "contexts": torch.tensor(context_tokens, dtype=torch.long),
            "centers": torch.tensor(center_token, dtype=torch.long),
        }
