import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class SeqContext2(nn.Module):

	def __init__(self, g_dim1, g_dim2, args):
		super(SeqContext2, self).__init__()
		self.input_size = g_dim1
		self.hidden_dim = g_dim2
		if args.rnn2 == "lstm":
			self.rnn2 = nn.LSTM(self.input_size, self.hidden_dim //2, dropout=args.drop_rate,  # // 取整除，向下取整
							   bidirectional=True, num_layers=1, batch_first=True)
		elif args.rnn2 == "gru":
			self.rnn2 = nn.GRU(self.input_size, self.hidden_dim // 2, dropout=args.drop_rate,
							  bidirectional=True, num_layers=1, batch_first=True)

	def forward(self, text_len_tensor, text_tensor):
		packed = pack_padded_sequence( # 将一个填充过的变长序列压紧
			text_tensor,
			text_len_tensor.cpu(),
			batch_first=True,
			enforce_sorted=False
		)
		rnn_out, (_, _) = self.rnn2(packed, None)
		# rnn_out, hn = self.rnn2(packed, None)
		rnn_out, _ = pad_packed_sequence(rnn_out, batch_first=True)	 # 重新padding

		return rnn_out
