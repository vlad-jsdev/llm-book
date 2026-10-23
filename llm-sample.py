import torch
import torch.nn as nn
import tiktoken
# import re
# import tiktoken
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
#
# query = inputs[1]
# attn_scores_2 = torch.empty(inputs.shape[0])
# attn_weights_2 = torch.softmax(attn_scores_2, dim=0) 
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
# attn_scores = inputs @ inputs.T
# # print(attn_scores)
#
# attn_weights = torch.softmax(attn_scores, dim=-1) 
# # print(attn_weights)
# all_context_vecs = attn_weights @ inputs
# print(all_context_vecs)


#
x_2 = inputs[1]
d_in = inputs.shape[1]
d_out = 2

torch.manual_seed(123)
W_query = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False) 
W_key = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False) 
W_value = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
query_2 = x_2 @ W_query
key_2 = x_2 @ W_key
value_2 = x_2 @ W_value
# print(query_2)
# class CausalAttention(nn.Module):
# 	def __init__(self, d_in, d_out, context_length, dropout, qkv_bias=False):
# 		super().__init__()
# 		self.d_out = d_out
# 		self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
# 		self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
# 		self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
# 		self.dropout = nn.Dropout(dropout)
# 		self.register_buffer(
# 			"mask", 
# 			torch.triu(torch.ones(context_length, context_length),
# 			diagonal=1)
# 		)
# 	def forward(self, x):
# 		b, num_tokens, d_in = x.shape
# 		keys = self.W_key(x)
# 		queries = self.W_query(x)
# 		values = self.W_value(x)
# 		attn_scores = queries @ keys.transpose(1, 2)
# 		attn_scores.masked_fill_(self.mask.bool() [:num_tokens, :num_tokens], -torch.inf)
# 		attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
# 		attn_weights = self.dropout(attn_weights)
# 		context_vec = attn_weights @ values
# 		return context_vec
#
# # print("context_vec.shape:", context_vec.shape)
#
# class MultuHeadAttention(nn.Module):
# 	def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
# 		super().__init__()
# 		self.heads = nn.ModuleList(
# 			[
# 				CausalAttention(d_in, d_out, context_length, dropout, qkv_bias)
# 				for _ in range(num_heads)
# 			]
# 		)
# 	def forward(self, x):
# 		return torch.cat([head(x) for head in self.heads], dim=-1)
#
class MultuHeadAttention(nn.Module):
	def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
		super().__init__()
		assert(d_out % num_heads == 0)
		self.d_out = d_out
		self.num_heads = num_heads
		self.head_dim = d_out // num_heads
		self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
		self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
		self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
		self.out_proj = nn.Linear(d_out, d_out)
		self.dropout = nn.Dropout(dropout)
		self.register_buffer(
			"mask", 
			torch.triu(torch.ones(context_length, context_length),
			diagonal=1)
		)
	def forward(self, x):
		b, num_tokens, d_in = x.shape
		keys = self.W_key(x)
		queries = self.W_query(x)
		values = self.W_value(x)
		keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
		values = values.view(b, num_tokens, self.num_heads, self.head_dim)
		queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)
		keys = keys.transpose(1, 2)
		queries = queries.transpose(1, 2)
		values = values.transpose(1, 2)
		attn_scores = queries @ keys.transpose(2, 3)
		mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
		attn_scores.masked_fill_(mask_bool, -torch.inf)
		attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
		attn_weights = self.dropout(attn_weights)
		context_vec = (attn_weights @ values).transpose(1, 2)
		context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
		context_vec = self.out_proj(context_vec)
		return context_vec
# a = torch.tensor([[[[0.2745, 0.6584, 0.2775, 0.8573],
# [0.8993, 0.0390, 0.9268, 0.7388], [0.7179, 0.7058, 0.9156, 0.4340]],
# [[0.0772, 0.3565, 0.1479, 0.5331], [0.4066, 0.2318, 0.4545, 0.9737], [0.4606, 0.5159, 0.4220, 0.5786]]]])

batch = torch.stack((inputs, inputs), dim=0) 
torch.manual_seed(123)
context_length = batch.shape[1]
d_in, d_out = 3, 8
mha = MultuHeadAttention(d_in, d_out, context_length, dropout=0.0, num_heads=4)
context_vecs = mha(batch)
print("context_vecs:", context_vecs.shape)
# first_head = a[0, 0, :, :]
# first_res = first_head @ first_head.T 
# print("First head:\n", first_res)
# second_head = a[0, 1, :, :] 
# second_res = second_head @ second_head.T 
# print("\nSecond head:\n", second_res)
# print("MHA:", mha)

tokenizer = tiktoken.get_encoding("gpt2")
batch = []
txt1 = "Every effort moves you"
txt2 = "Every day holds a"
batch.append(torch.tensor(tokenizer.encode(txt1))) 
batch.append(torch.tensor(tokenizer.encode(txt2))) 
batch = torch.stack(batch, dim=0)
print(batch)
