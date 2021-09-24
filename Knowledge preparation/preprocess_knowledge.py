import csv
import argparse
import pickle
import numpy as np, pandas as pd
from ast import literal_eval
from collections import defaultdict
#from utils.io import load_pickle, to_pickle
from nltk.tokenize import word_tokenize,regexp_tokenize,wordpunct_tokenize,blankline_tokenize
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
import os
import nltk
from nltk.corpus import stopwords
from numpy import genfromtxt
from nltk.stem import WordNetLemmatizer
nltk.download('stopwords')
nltk.download('wordnet')
wlem=WordNetLemmatizer()
stoplist=stopwords.words('english')

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.9
session = tf.Session(config=config)
KTF.set_session(session)

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

def get_all_ngrams(examples):
	num=0
	sum=0
	m=0
	data_ngrams = []
	for ex in examples:
		flag=0
		each_ngrams=[]
		for utter in ex:
			if utter.lower() not in stoplist:
				num=num+1
				each_ngrams.append(wlem.lemmatize(utter.lower()))
				flag=1
		if flag==1:
			sum+=1
		else:
			m+=1
			for n in ex:
				each_ngrams.append(n)
		data_ngrams.append(each_ngrams)
	print("data_token:",num,sum,m)  #27667 5531
	print(len(np.array(data_ngrams)),data_ngrams[5])  # set 无序不重复元素集
	return (data_ngrams)


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--dataset', required=False)
	parser.add_argument('--n', default=1)
	args = parser.parse_args()

	dataset = args.dataset
	n = args.n

	print("Loading dataset...")
	data=[]
	with open('./Transcripts_context.txt', 'r') as f:
		contexts=f.readlines()
		for each in contexts:
			sens = each.strip()
			b=regexp_tokenize(sens, pattern='\w+')
			data.append(b)
	print(np.array(data).shape)

	ngrams = get_all_ngrams(data)
	
	# load unconceptnet txt
	unconceptnet=[]
	f=open('./unconceptnet.txt',encoding='utf-8')
	lines=f.readlines()
	for line in lines:
		line=line.strip().split()
		unconceptnet.append(line)
	print(np.array(unconceptnet))

	#################### Part1: prepare conceptnet ###########################
	f=open ('./numberbatch-en-19.08.txt/numberbatch-en.txt',encoding='utf-8')
	lines=f.readlines()
	rows=len(lines)
	datamat=[]

	row=0
	for line in lines:
		line=line.strip().split()
		if(line[0][0]!='#'):
			datamat.append(line)
			row+=1
	print(row,datamat[1000])  # row=516638

	gram=0
	sum=0
	num=0
	concept_data=[]
	max=0
	print("len(ngrams):",len(ngrams),len(ngrams[5]))  
	for i in range(0,len(ngrams)):
		data_each=[]
		flag1=0
		kk=0
		for j in range(len(ngrams[i])):
			flag=0
			kk+=1
			for m in range(0,len(datamat)):
				if (ngrams[i][j] == datamat[m][0]):
					data_each.append(datamat[m][1:])
					gram=gram+1
					flag=1
					flag1=1
					break
			if flag==0:
				for n in range(len(unconceptnet)):
					if(ngrams[i][j]==unconceptnet[n][0]):
						data_each.append(unconceptnet[n][1:])
						num+=1
						flag1=1
						break
		if kk>max:
			max=kk
		if flag1==1:
			sum+=1
		concept_data.append(data_each)			
	print("concept:",gram,num,len(concept_data),np.array(concept_data[5]).shape) #gram=27639 5531
	print("sum:",sum)   #27639 5316
	print("max num of each sentences:",max)
	
	################### Part2: prepare lexicon   ##########################################
	Anew = pd.read_csv('./NRC_VAD_Lexicon.csv', delimiter=',',header=None)
	Anew=np.asarray(Anew)
	
	Anew_num=0
	Anew_data=[]
	for i in range(0,len(ngrams)):
		data_each=[]
		for j in range(len(ngrams[i])):
			flag=0
			for k in range(0, len(Anew)):
				if (ngrams[i][j] == Anew[k][0]):
					Anew_num=Anew_num+1
					data_each.append(Anew[k][1:])
					flag=1
			if(flag==0):
				#vector=np.asarray([5,1,5])
				vector=np.asarray([0.5,0.5,0.5])
				data_each.append(vector)
		Anew_data.append(data_each)
	print("Anew:",Anew_num,len(Anew)) # Anew:16820   NRC:18561

	# prepare context information
	Dialogue_label=pd.read_csv('./Dialogue_label.csv',header=None)
	print("type:",type(Dialogue_label))
	Dialogue_label=np.asarray(Dialogue_label)
	print("type:",type(Dialogue_label))
	train_Dialogue = Dialogue_label[:4290]
	test_Dialogue= Dialogue_label[4290:5531]
	
	print(test_Dialogue)

	train_video_mapping=defaultdict(list)
	test_video_mapping=defaultdict(list)

	m=0
	n=0
	for i in range(train_Dialogue.shape[0]):
		m=m+1
		train_video_mapping[train_Dialogue[i][0]].append(concept_data[i])
	
	for j in range(test_Dialogue.shape[0]):
		n=n+1
		test_video_mapping[test_Dialogue[j][0]].append(concept_data[4290+j])
	print(m,n)
	
	Concept_data_train=[]
	Concept_data_test=[]
	Anew_data_train=[]
	Anew_data_test=[]
	count=0

	for key,value in train_video_mapping.items():
		con_train, anew_train=[], []
		ctr=0
		for i in range(len(value)):
			ctr=ctr+1
			con_train.append(concept_data[i+count])
			anew_train.append(Anew_data[i+count])
		count+=ctr
		Concept_data_train.append(con_train)
		Anew_data_train.append(anew_train)
	print("train count:",count)
	
	#count=0
	for key,value in test_video_mapping.items():
		con_test, anew_test =[], []
		ctr=0
		for i in range(len(value)):
			ctr=ctr+1
			con_test.append(concept_data[i+count])
			anew_test.append(Anew_data[i+count])
		count+=ctr
		Concept_data_test.append(con_test)
		Anew_data_test.append(anew_test)
	print("test count:",count)
			
	Concept_data_train=np.asarray(Concept_data_train)
	Concept_data_test=np.asarray(Concept_data_test)
	Anew_data_train=np.asarray(Anew_data_train)
	Anew_data_test=np.asarray(Anew_data_test)
	
	print(np.array(Concept_data_train).shape,np.array(Concept_data_train[2]).shape,np.array(Concept_data_train[2][2]).shape)
	print("Dumping Knowledge data:")
	with open('./knowledge_nrc.pickle','wb') as handle:
		pickle.dump((Concept_data_train,Concept_data_test,Anew_data_train,Anew_data_test), handle, protocol=pickle.HIGHEST_PROTOCOL)
	
