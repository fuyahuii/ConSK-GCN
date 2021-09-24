import torch
import torch.nn as nn

from .SeqContext1 import SeqContext1
from .SeqContext2 import SeqContext2
from .KG_EdgeAtt_new import KG_EdgeAtt_new
from .GCN import GCN
from .Classifier import Classifier
from .functions import batch_graphify
import dgcn

log = dgcn.utils.get_logger()


class CONSKGCN(nn.Module):

	def __init__(self, args):
		super(CONSKGCN, self).__init__()
		u_dim = 612	 # Concatenated multimodal features
		g_dim1 = 1224	# GRU feature, Context-aware utterance representation
		g_dim2=300
		h1_dim = 100 # hidden size
		h2_dim = 100 # hidden size
		hc_dim = 100 # hidden size
		tag_size = 4 # classes

		self.wp = args.wp
		self.wf = args.wf
		self.device = args.device

		self.rnn1 = SeqContext1(u_dim, g_dim1, args)
		self.rnn2= SeqContext2(g_dim1, g_dim2, args)
		
		self.edge_att = KG_EdgeAtt_new(g_dim2, args)
		
		self.gcn = GCN(g_dim2, h1_dim, h2_dim, args)
		self.clf = Classifier(h2_dim+g_dim2, hc_dim, tag_size, args)

		edge_type_to_idx = {}
		for j in range(args.n_speakers):
			for k in range(args.n_speakers):
				edge_type_to_idx[str(j) + str(k) + '0'] = len(edge_type_to_idx)
				edge_type_to_idx[str(j) + str(k) + '1'] = len(edge_type_to_idx)
		self.edge_type_to_idx = edge_type_to_idx
		log.debug(self.edge_type_to_idx)

	def get_rep(self, data):
		
		train_data=torch.cat((data["train_audio"], data["train_text"]), dim=-1).to(self.device)
		
		lstm_features = self.rnn1(data["train_len_tensor"], train_data) # [batch_size, mx_len, D_g]
		node_features=self.rnn2(data["train_len_tensor"], lstm_features)

		features, edge_index, edge_norm, edge_type, edge_index_lengths = batch_graphify(
			node_features, data["train_len_tensor"], data["speaker_tensor"], data["knowledge_tensor"],data["anew_tensor"],self.wp, self.wf,
			self.edge_type_to_idx, self.edge_att, self.device)

		graph_out = self.gcn(features, edge_index, edge_norm, edge_type)
		
		return graph_out, features

	def forward(self, data):
		graph_out, features= self.get_rep(data)
		out = self.clf(torch.cat([graph_out, features], dim=-1), data["train_len_tensor"])

		return out

	def get_loss(self, data):
		graph_out, features = self.get_rep(data)
		loss = self.clf.get_loss(torch.cat([graph_out, features], dim=-1),
								 data["label_tensor"], data["train_len_tensor"])
		return loss
