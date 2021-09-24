import math
import random

import torch
import numpy as np

class Dataset:

	def __init__(self, samples_text,samples_audio,labels,length, knowledge,anew,batch_size):
		self.samples_text = samples_text
		self.samples_audio=samples_audio
		self.labels=labels
		self.text_length=length
		self.batch_size = batch_size
		self.knowledge=knowledge
		self.anew=anew
		if self.samples_audio is not None:
			self.num_batches = math.ceil(len(self.samples_audio) / batch_size)
		elif self.samples_text is not None:
			self.num_batches = math.ceil(len(self.samples_text) / batch_size)

	def __len__(self):
		return self.num_batches

	def __getitem__(self, index):
		batch_text,batch_audio, batch_label, batch_text_length,batch_knowledge,batch_anew = self.raw_batch(index)

		return self.padding(batch_text,batch_audio, batch_label, batch_text_length,batch_knowledge,batch_anew)

	def raw_batch(self, index):
		assert index < self.num_batches, "batch_idx %d > %d" % (index, self.num_batches)
		batch_audio, batch_text = None, None
		if self.samples_audio is not None:
			batch_audio = self.samples_audio[index * self.batch_size: (index + 1) * self.batch_size]
		if self.samples_text is not None:
			batch_text = self.samples_text[index * self.batch_size: (index + 1) * self.batch_size]
		batch_label=self.labels[index * self.batch_size: (index + 1) * self.batch_size]
		#batch_speaker=self.speakers[index * self.batch_size: (index + 1) * self.batch_size]
		batch_text_length=self.text_length[index*self.batch_size:(index+1)*self.batch_size]
		batch_knowledge=self.knowledge[index*self.batch_size:(index+1)*self.batch_size]
		batch_anew=self.anew[index*self.batch_size:(index+1)*self.batch_size]

		return batch_text,batch_audio, batch_label,batch_text_length,batch_knowledge,batch_anew

	def padding(self, batch_text,batch_audio,batch_label, text_len_tensor,batch_knowledge,batch_anew):
		batch_size = len(batch_audio) if batch_audio is not None else len(batch_text)
		# text_len_tensor = torch.tensor(text_len_tensor).long()
		#speaker_tensor = None
		labels = []
		# *********************** rnn for text data****************************
		if batch_text is not None:
			global train_text, train_len_text
			dim_t = len(batch_text[0][0])
			# print("*******text dim_t", dim_t)
			train_len_text = torch.tensor([len(s) for s in batch_text]).long()
			mx = torch.max(train_len_text).item()

			train_text = torch.zeros((batch_size, mx, dim_t))
			for i, s in enumerate(batch_text):
				cur_len = len(s)
				tmp = [torch.from_numpy(t).float() for t in batch_text[i]]
				tmp = torch.stack(tmp)
				train_text[i, :cur_len, :] = tmp

		# *********************** rnn for audio data*************************
		# 改变说话人信息的存储方式
		# speaker_tensor = torch.zeros((batch_size, mx)).long()
		if batch_audio is not None:
			global train_len_audio, train_audio
			train_len_audio = torch.tensor([len(s) for s in batch_audio]).long()
			mx = torch.max(train_len_audio).item()
			dim_t = len(batch_audio[0][0])
			# print("*******audio dim_t", dim_t)
			train_audio = torch.zeros((batch_size, mx, dim_t))

			#speaker_tensor = torch.zeros((batch_size, mx)).long()

			for i, s in enumerate(batch_audio):
				cur_len = len(s)
				tmp = [torch.from_numpy(t).float() for t in batch_audio[i]]
				tmp = torch.stack(tmp)
				train_audio[i, :cur_len, :] = tmp
				#speaker_tensor[i, :cur_len] = torch.tensor([self.speaker_to_idx[c] for c in batch_speaker[i]])
				# speaker_tensor.extend([self.speaker_to_idx[c] for c in batch_speaker[i]])
				labels.extend(batch_label[i])

		# *******************************knowledge tensor preparation*************************
		mx_k = 0
		for i, s in enumerate(batch_knowledge):
			knowledge_len = torch.tensor([len(s) for s in batch_knowledge[i]]).long()
			# print("*****",knowledge_len)
			mx1 = torch.max(knowledge_len).item()
			mx_k = max(mx1, mx_k)
		# print("mk:",mx_k)

		knowledge_tensor = torch.zeros((batch_size, mx, mx_k, 300))
		# knowledge_tensor=torch.zeros((batch_size,110,mx_k,300))
		for i, s in enumerate(batch_knowledge):
			cur_len = len(s)
			for j, m in enumerate(batch_knowledge[i]):
				cur_len_k = len(m)
				# print(type(batch_knowledge[i][j]),np.array(batch_knowledge[i][j]))
				for k, word in enumerate(batch_knowledge[i][j]):
					tem = [float(t) for t in word]
					tem = torch.Tensor(tem)
					# tem=[torch.from_numpy(t).float() for t in np.array(word)]
					knowledge_tensor[i, j, k, :] = tem
		# print("knowledge_tensor:",knowledge_tensor.shape)
		# *****************************anew lexicon tensor preparation***********************
		mx_a = 0
		for i, s in enumerate(batch_anew):
			anew_len = torch.tensor([len(s) for s in batch_anew[i]]).long()
			mx2 = torch.max(anew_len).item()
			mx_a = max(mx2, mx_a)

		anew_tensor = torch.zeros((batch_size, mx, mx_a, 3))

		for i, s in enumerate(batch_anew):
			for j, m in enumerate(batch_anew[i]):
				for k, word in enumerate(batch_anew[i][j]):
					tem = [float(t) for t in word]
					tem = torch.Tensor(tem)
					# tem=[torch.from_numpy(t).float() for t in np.array(word)]
					anew_tensor[i, j, k, :] = tem
		# tem=[torch.Tensor(t).float() for t in batch_anew[i][j]]
		# tem.torch.stack(tem)
		# anew_tensor[i,j,:cur_len_k,:]=tem

		#speaker_label = torch.tensor(speaker_tensor).long()
		label_tensor = torch.tensor(labels).long()

		data = {
			"train_len_tensor": train_len_audio,
			# "train_len_text": train_len_text,
			"train_text": train_text,
			"train_audio": train_audio,
			#"speaker_tensor": speaker_label,
			"label_tensor": label_tensor,
			"knowledge_tensor": knowledge_tensor,
			"anew_tensor": anew_tensor
		}
		return data


	def shuffle(self):
		random.shuffle(self.samples)




