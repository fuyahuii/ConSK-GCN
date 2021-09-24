import numpy as np
import torch

import dgcn

log = dgcn.utils.get_logger()


def batch_graphify(features, lengths, knowledge,anew,wp, wf,  att_model, device):
	node_features, edge_index, edge_norm, edge_type = [], [], [], []
	batch_size = features.size(0)
	length_sum = 0
	edge_ind = []
	edge_index_lengths = []

	for j in range(batch_size):
		edge_ind.append(edge_perms(lengths[j].cpu().item(), wp, wf))

	edge_weights = att_model(features, lengths,knowledge,anew, edge_ind)

	for j in range(batch_size):
		cur_len = lengths[j].item() # .item()是得到一个元素张量里面的元素值
		node_features.append(features[j, :cur_len, :])
		perms = edge_perms(cur_len, wp, wf)
		perms_rec = [(item[0] + length_sum, item[1] + length_sum) for item in perms]
		length_sum += cur_len
		edge_index_lengths.append(len(perms))

		for item, item_rec in zip(perms, perms_rec):
			edge_index.append(torch.tensor([item_rec[0], item_rec[1]]))
			edge_norm.append(edge_weights[j][item[0], item[1]])
			# edge_norm.append(edge_weights[j, item[0], item[1]])

	node_features = torch.cat(node_features, dim=0).to(device)	# [E, D_g]
	edge_index = torch.stack(edge_index).t().contiguous().to(device)  # [2, E]
	edge_norm = torch.stack(edge_norm).to(device)  # [E]
	edge_type = torch.tensor(edge_type).long().to(device)  # [E]
	edge_index_lengths = torch.tensor(edge_index_lengths).long().to(device) # [B]
	# print(len(edge_index),edge_index)
	return node_features, edge_index, edge_norm, edge_index_lengths


def edge_perms(length, window_past, window_future):
	"""
	Method to construct the edges of a graph (a utterance) considering the past and future window.
	return: list of tuples. tuple -> (vertice(int), neighbor(int))
	"""

	all_perms = set()
	array = np.arange(length)
	for j in range(length):
		perms = set()

		if window_past == -1 and window_future == -1:
			eff_array = array
		elif window_past == -1:	 # use all past context
			eff_array = array[:min(length, j + window_future + 1)]
		elif window_future == -1:  # use all future context
			eff_array = array[max(0, j - window_past):]
		else:
			eff_array = array[max(0, j - window_past):min(length, j + window_future + 1)]

		for item in eff_array:
			perms.add((j, item)) # add()给集合添加元素，如果添加的元素在集合中已存在，则不执行任何操作
		all_perms = all_perms.union(perms)	# union()返回两个集合的并集

	return list(all_perms)
