import torch
import torch.nn as nn

from .SeqContext import SeqContext
from .KG_EdgeAtt_new import KG_EdgeAtt_new
from .GCN import GCN
from .Classifier import Classifier
from .functions import batch_graphify
import dgcn

log = dgcn.utils.get_logger()


class CONSKGCN(nn.Module):

	def __init__(self, args):
		super(CONSKGCN, self).__init__()
		ut_dim = 768	 # Context-independent Text/Audio utterance
		ua_dim=512
		gt_dim1 = 768    # GRU feature, Context-aware utterance representation
		ga_dim1=512

		h1_dim = args["gcn_h1"]  # hidden size
		h2_dim = args["gcn_h2"]  # hidden size
		hc_dim = 100 # hidden size
		tag_size = 7 # classes

		self.wp = args["wp"]
		self.wf = self.wp
		self.device = args["device"]
		self.param_t = args["param_t"]
		self.param_a = self.param_t
		
		self.rnn_t = SeqContext(ut_dim, gt_dim1, args)
		self.rnn_a = SeqContext(ua_dim, ga_dim1, args)
		self.edge_att_t = KG_EdgeAtt_new(gt_dim1, args,self.param_t)
		self.edge_att_a = KG_EdgeAtt_new(ga_dim1, args,self.param_a)
		self.gcn_t = GCN(gt_dim1, h1_dim, h2_dim, args)
		self.gcn_a = GCN(ga_dim1, h1_dim, h2_dim, args)

		self.clf = Classifier(gt_dim1+ga_dim1+2*h2_dim,hc_dim, tag_size, args)

	def get_rep(self, data):

		node_features_t = self.rnn_t(data["train_len_tensor"], data["train_text"]) # [batch_size, mx_len, D_g]
		node_features_a = self.rnn_a(data["train_len_tensor"], data["train_audio"])  # [batch_size, mx_len, D_g]

		#Text ConSK-GCN construction
		features_t, edge_index_t, edge_norm_t,  edge_index_lengths_t = batch_graphify(
			node_features_t, data["train_len_tensor"],  data["knowledge_tensor"],data["anew_tensor"],self.wp, self.wf,
			self.edge_att_t, self.device)

		graph_out_t = self.gcn_t(features_t, edge_index_t, edge_norm_t, edge_type=None)

		#Audio ConSK-GCN construction
		features_a, edge_index_a, edge_norm_a, edge_index_lengths_a = batch_graphify(
			node_features_a, data["train_len_tensor"], data["knowledge_tensor"], data["anew_tensor"], self.wp, self.wf,
			self.edge_att_a, self.device)

		graph_out_a = self.gcn_a(features_a, edge_index_a, edge_norm_a, edge_type=None)

		graph_out=torch.cat((graph_out_t,graph_out_a),dim=-1).to(self.device)
		lstm_features = torch.cat((features_a,features_t), dim=-1).to(self.device)

		return graph_out,lstm_features

	def forward(self, data):
		graph_out,features= self.get_rep(data)
		out = self.clf(torch.cat([graph_out, features], dim=-1), data["train_len_tensor"])

		return out

	def get_loss(self, data):
		graph_out,features= self.get_rep(data)
		loss = self.clf.get_loss(torch.cat([graph_out, features], dim=-1),
								 data["label_tensor"], data["train_len_tensor"])
		return loss
