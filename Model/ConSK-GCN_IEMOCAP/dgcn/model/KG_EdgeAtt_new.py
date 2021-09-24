import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math 

import dgcn

log = dgcn.utils.get_logger()

class ScaledDotProductAttention(nn.Module):

	def forward(self, query, key, value, mask=None):
		dk = query.size()[-1]
		scores = query.matmul(key.transpose(-2, -1)) / math.sqrt(dk)
		if mask is not None:
			scores = scores.masked_fill(mask == 0, -1e9)
		attention = F.softmax(scores, dim=-1)
		print("ScaledDotProductAttention")
		return attention.matmul(value)


class MultiHeadAttention(nn.Module):
	def __init__(self,
				 in_features,
				 head_num,
				 bias=True,
				 activation=F.relu):
		"""Multi-head attention.
		
		:param in_features: Size of each input sample.
		:param head_num: Number of heads.
		:param bias: Whether to use the bias term.
		:param activation: The activation after each linear transformation.
		"""
		super(MultiHeadAttention, self).__init__()
		if in_features % head_num != 0:
			raise ValueError('`in_features`({}) should be divisible by `head_num`({})'.format(in_features, head_num))
		self.in_features = in_features
		self.head_num = head_num
		self.activation = activation
		self.bias = bias
		self.linear_q = nn.Linear(in_features, in_features, bias)
		self.linear_k = nn.Linear(in_features, in_features, bias)
		self.linear_v = nn.Linear(in_features, in_features, bias)
		self.linear_o = nn.Linear(in_features, in_features, bias)

	def forward(self, q, k, v, mask=None):
		q, k, v = self.linear_q(q), self.linear_k(k), self.linear_v(v)
		if self.activation is not None:
			q = self.activation(q)
			k = self.activation(k)
			v = self.activation(v)

		q = self._reshape_to_batches(q)
		k = self._reshape_to_batches(k)
		v = self._reshape_to_batches(v)
		#mask=self.gen_history_mask(q)
		if mask is not None:
			mask = mask.repeat(self.head_num, 1, 1)
		y = ScaledDotProductAttention()(q, k, v, mask)
		y = self._reshape_from_batches(y)

		y = self.linear_o(y)
		if self.activation is not None:
			y = self.activation(y)
		print("multihead")
		return y

	@staticmethod
	def gen_history_mask(x):
		"""Generate the mask that only uses history data.

		:param x: Input tensor.
		:return: The mask.
		"""
		batch_size, seq_len, _ = x.size()
		return torch.tril(torch.ones(seq_len, seq_len)).view(1, seq_len, seq_len).repeat(batch_size, 1, 1)

	def _reshape_to_batches(self, x):
		batch_size, seq_len, in_feature = x.size()
		sub_dim = in_feature // self.head_num
		return x.reshape(batch_size, seq_len, self.head_num, sub_dim)\
				.permute(0, 2, 1, 3)\
				.reshape(batch_size * self.head_num, seq_len, sub_dim)

	def _reshape_from_batches(self, x):
		batch_size, seq_len, in_feature = x.size()
		batch_size //= self.head_num
		out_dim = in_feature * self.head_num
		return x.reshape(batch_size, self.head_num, seq_len, in_feature)\
				.permute(0, 2, 1, 3)\
				.reshape(batch_size, seq_len, out_dim)

	def extra_repr(self):
		return 'in_features={}, head_num={}, bias={}, activation={}'.format(
			self.in_features, self.head_num, self.bias, self.activation,
		)

class KG_EdgeAtt_new(nn.Module):

	def __init__(self, g_dim,args):
		super(KG_EdgeAtt_new, self).__init__()
		self.device = args.device
		self.wp = args.wp
		self.wf = args.wf
		
		# semantic weights
		self.weight_sem=nn.Parameter(torch.zeros((g_dim,g_dim)).float(),requires_grad=True)
		var = 2. / (self.weight_sem.size(0) + self.weight_sem.size(1))
		self.weight_sem.data.normal_(0, var) # 正态分�?		
		# conceptnet weights
		self.weight_con = nn.Parameter(torch.zeros((300,300)).float(), requires_grad=True)
		var = 2. / (self.weight_con.size(0) + self.weight_con.size(1))
		self.weight_con.data.normal_(0, var) # 正态分�?
		
		# conceptnet weights
		self.weight = nn.Parameter(torch.zeros((110,110)).float(), requires_grad=True)
		var = 2. / (self.weight.size(0) + self.weight.size(1))
		self.weight.data.normal_(0, var) # 正态分�?

	def forward(self, node_features, text_len_tensor, knowledge,anew,edge_ind):
		batch_size, mx_len = node_features.size(0), node_features.size(1)
		num_len=knowledge.size(2)
		weight_sem = self.weight_sem.unsqueeze(0).unsqueeze(0)
		att_matrix_sem = torch.matmul(weight_sem, node_features.unsqueeze(-1)).squeeze(-1)	# [B, L, D_g]

		alphas_sem = torch.zeros((batch_size,mx_len, 110)).to(self.device)
		for i in range(batch_size):
			cur_len = text_len_tensor[i].item()
			# ***************edge weights of semantic graph*********************************
			for j in range(cur_len):
				s = j - self.wp if j - self.wp >= 0 else 0
				e = j + self.wf if j + self.wf <= cur_len - 1 else cur_len - 1
				tmp = att_matrix_sem[i, s: e + 1, :]	# [L', D_g]
				feat = node_features[i, j]	# [D_g]
				#score = torch.matmul(tmp, feat)
				score=1-torch.acos(torch.cosine_similarity(feat,tmp,dim=-1))/math.pi
				probs = F.softmax(score)	# [L']
				alphas_sem[i,j, s: e + 1] = probs
		
		# ***************edge wights of affective enriched knowledge graph*****************
		anew_aff=torch.zeros((batch_size,mx_len,num_len)).to(self.device)
		for i in range(batch_size):
			for j in range(mx_len):
				for n in range(len(anew[i][j])):
					v=anew[i][j][n][0]
					a=anew[i][j][n][0]
					a=a/2
					aff1=torch.norm(torch.sub(torch.Tensor([v, a]),torch.Tensor([0.5, 0])))
					aff2=torch.sub(aff1,0.06467)
					aff3=torch.div(aff2,0.607468)
					anew_aff[i,j,n]=aff3

		weight_con=self.weight_con.unsqueeze(0).unsqueeze(0)
		att_matrix_con=torch.matmul(knowledge,weight_con)
		att_matrix_con_aff=torch.mul(att_matrix_con,anew_aff.unsqueeze(-1))
		
		#alphas_con=[]
		alphas_con=torch.zeros((batch_size,mx_len,110)).to(self.device)
		for i in range(batch_size):
			cur_len = text_len_tensor[i].item()
			for j in range(cur_len):
				s = j - self.wp if j - self.wp >= 0 else 0
				e = j + self.wf if j + self.wf <= cur_len - 1 else cur_len - 1
				tmp = knowledge[i, j,:,:]
				feat=att_matrix_con_aff[i,s: e + 1,:,:]
				#score=torch.sum(1-torch.acos(torch.cosine_similarity(feat,tmp,dim=-1))/math.pi,dim=1)
				score=torch.sum(torch.abs(torch.cosine_similarity(feat,tmp,dim=-1)),dim=1)
				#probs = F.softmax(score)	# [L']
				alphas_con[i,j, s: e + 1] = 10*score
		
		#***************knowledge enriched attention weights***********
		knowledge_att=0.5*alphas_sem+0.5*alphas_con
		#knowledge_att=torch.mul(alphas_con,alphas_sem)

		return knowledge_att



