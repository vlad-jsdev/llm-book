import re
import tiktoken
import torch
#
# with open("the-verdict.txt", "r", encoding="utf-8") as f:
#     raw_text = f.read()
# # print("Total number of character:", len(raw_text))
# # print(raw_text[:99])
#
# # text = "Hello, world. Is this-- a test?"
# preprocessed = re.split(r'([,.:?_!"()\']|--|\s)', raw_text)
# preprocessed = [item.strip() for item in preprocessed if item.strip()]
#
# # all_words = sorted(list(set(preprocessed)))
# all_tokens = sorted(list(set(preprocessed)))
#
# vocab_size = len(all_tokens)
# all_tokens.extend(["<|endoftext|>", "<|unk|>"])
# vocab = {token: integer for integer, token in enumerate(all_tokens)}
# # for i, item in enumerate(vocab.items()):
# #     print(item)
# #     if i >= 50:
# #         break
#
# # print(len(vocab.items()))
# # for i, item in enumerate(list(vocab.items())[-5:]):
# #     print(item)
#
# class SimpleTokenizerV2:
#     def __init__(self, vocab):
#         self.str_to_int = vocab 
#         self.int_to_str = {i:s for s,i in vocab.items()}
#
#     def encode(self, text):
#         preprocessed = re.split(r'([,.:?_!"()\']|--|\s)', text)
#         preprocessed = [
#             item.strip() for item in preprocessed if item.strip()
#         ]
#         preprocessed = [item if item in self.str_to_int
#                         else "<|unk|>" for item in preprocessed]
#         ids = [self.str_to_int[s] for s in preprocessed]
#         return ids
#
#     def decode(self, ids):
#         text = " ".join([self.int_to_str[i] for i in ids])
#
#         text = re.sub(r'\s+([,.:;?!"()\'])', r'\1', text)
#         return text
#
# # tokenizer = SimpleTokenizerV2(vocab)
# # text1 = "Hello, do you like tea?"
# # text2 = "In the sunlit terraces of the palace."
# # text = " <|endoftext|> ".join((text1, text2))
# # ids = tokenizer.encode(text)
# # # print(tokenizer.encode(text))
# #
# # print(tokenizer.decode(ids))
#
# tokenizer = tiktoken.get_encoding("gpt2") 
#
# with open("the-verdict.txt", "r", encoding="utf-8") as f:
#     raw_text = f.read()
#
# # text = (
# # "Hello, do you like tea? <|endoftext|> In the sunlit terraces"
# # "of someunknownPlace."
# # )
# # integers = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
# # print(integers)
#
# enc_text = tokenizer.encode(raw_text)
# enc_sample = enc_text[50:]
#
# content_size = 4
# x = enc_sample[:content_size]
# y = enc_sample[1:content_size+1]
#
# for i in range(1, content_size+1):
#     context = enc_sample[:i]
#     desired = enc_sample[i]
#     print(tokenizer.decode(context), "---->", tokenizer.decode([desired]))
# # print(f"x: {x}")
# # print(f"y: {y}")
# # print(len(enc_sample))
#
#

inputs = torch.tensor(
	[[0.43, 0.15, 0.89], # Your (x^1)
	[0.55, 0.87, 0.66], # journey (x^2)
	[0.57, 0.85, 0.64], # starts (x^3)
	[0.22, 0.58, 0.33], # with (x^4)
	[0.77, 0.25, 0.10], # one (x^5)
	[0.05, 0.80, 0.55]] # step (x^6#
)

query = inputs[1]
attn_scores_2 = torch.empty(inputs.shape[0])
attn_weights_2 = torch.softmax(attn_scores_2, dim=0) 
# print("Attention weights:", attn_weights_2) 
# print("Sum:", attn_weights_2.sum())
# for i, x_i in enumerate(inputs):
# 	attn_scores_2[i] = torch.dot(x_i, query)
# print(attn_scores_2)
# attn_weights_2_tmp = attn_scores_2 / attn_scores_2.sum()
# print("Attention weights:", attn_weights_2_tmp)
# print("Sum:", attn_weights_2_tmp.sum())
# attn_scores = torch.empty(6, 6)
# for i, x_i in enumerate(inputs):
# 	for j, x_j in enumerate(inputs):

# 		attn_scores[i, j] = torch.dot(x_i, x_j) 
attn_scores = inputs @ inputs.T
# print(attn_scores)

attn_weights = torch.softmax(attn_scores, dim=-1) 
# print(attn_weights)
all_context_vecs = attn_weights @ inputs
print(all_context_vecs)
