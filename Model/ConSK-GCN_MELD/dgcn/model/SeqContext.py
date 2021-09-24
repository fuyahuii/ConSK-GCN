import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class SeqContext(nn.Module):

	def __init__(self, u_dim, g_dim, args):
		super(SeqContext, self).__init__()
		self.input_size = u_dim
		self.hidden_dim = g_dim
		print(args["rnn1"])
		if args["rnn1"] == "lstm":
			self.rnn1 = nn.LSTM(self.input_size, self.hidden_dim //2, dropout=args["drop_rate"],
							   bidirectional=True, num_layers=2, batch_first=True)
		elif args["rnn1"] == "gru":
			self.rnn1 = nn.GRU(self.input_size, self.hidden_dim // 2, dropout=args["drop_rate"],
							  bidirectional=True, num_layers=1, batch_first=True)

	def forward(self, text_len_tensor, text_tensor):
		packed = pack_padded_sequence( # Compress a padded variable-length sequence
			text_tensor,
			text_len_tensor.cpu(),
			batch_first=True,
			enforce_sorted=False
		)
		rnn_out, (_, _) = self.rnn1(packed, None)
		rnn_out, _ = pad_packed_sequence(rnn_out, batch_first=True)	 # repadding

		return rnn_out
