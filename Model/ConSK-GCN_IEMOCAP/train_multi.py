import argparse

import torch
import pickle
import dgcn
import numpy as np
import os
from tqdm import tqdm
import time
import logging

log = dgcn.utils.get_logger()

def setup_logging(log_file='log.txt'):
	"""
	Setup logging configuration
	"""
	logging.basicConfig(level=logging.INFO,
						format="%(asctime)s - %(levelname)s - %(message)s",	  
						datefmt="%Y-%m-%d %H:%M:%S",  
						filename=log_file,
						filemode='w')  

	console = logging.StreamHandler()
	console.setLevel(logging.INFO)
	formatter = logging.Formatter('%(message)s')
	console.setFormatter(formatter)
	logging.getLogger('').addHandler(console)
	
def main(args):
	dgcn.utils.set_seed(args.seed)

	# load data
	log.info("Loaded data.")
	with open('./Audio_IEMOCAP.pickle', 'rb') as handle:
		(train_audio, train_label, train_speaker, test_audio, test_label, test_speaker, maxlen, train_length, test_length) = pickle.load(handle)
	with open('./Text_IEMOCAP.pickle', 'rb') as handle:
		(train_text, train_label, train_speaker, test_text, test_label, test_speaker, maxlen, train_length, test_length) = pickle.load(handle)

	with open('./Knowledge_nrc_IEMOCAP.pickle','rb')as handle:
		(Concept_data_train,Concept_data_test,Anew_data_train,Anew_data_test)=pickle.load(handle)
			
	trainset = dgcn.Dataset(train_audio,train_text,train_label,train_speaker,train_length, Concept_data_train, Anew_data_train, args.batch_size)
	testset = dgcn.Dataset(test_audio,test_text,test_label, test_speaker,test_length, Concept_data_test, Anew_data_test, args.batch_size)

	log.debug("Building model...")
	
	model = dgcn.CONSKGCN(args).to(args.device)
	opt = dgcn.Optim(args.learning_rate, args.max_grad_value, args.weight_decay)
	opt.set_parameters(model.parameters(), args.optimizer)

	coach = dgcn.Coach(trainset, testset, model, opt,args) # Calculate f1 and loss

	# Train.
	log.info("Start training...")
	ret = coach.train()

	# Save.
	checkpoint = {
		"best_dev_f1": ret[0],
		"best_epoch": ret[1],
		"best_state": ret[2],
	}
	torch.save(checkpoint, model_file) # entire net


if __name__ == "__main__":

	save_path="./"
	setup_logging(os.path.join(save_path, 'log_test_ConSK-GCN_0.0001.txt'))
	
	parser = argparse.ArgumentParser(description="train.py")
	parser.add_argument("--data", type=str,default=False,
						help="Path to data")

	# Training parameters
	parser.add_argument("--from_begin", action="store_true", default=True,
						help="Training from begin.")
	parser.add_argument("--device", type=str, default="cuda",
						help="Computing device.")
	parser.add_argument("--epochs", default=60, type=int,
						help="Number of training epochs.")
	parser.add_argument("--batch_size", default=32, type=int,
						help="Batch size.")
	parser.add_argument("--optimizer", type=str, default="adam",
						choices=["sgd", "rmsprop", "adam"],
						help="Name of optimizer.")
	parser.add_argument("--learning_rate", type=float, default=0.0001,
						help="Learning rate.")
	parser.add_argument("--weight_decay", type=float, default=1e-8,
						help="Weight decay.")
	parser.add_argument("--max_grad_value", default=-1, type=float,
						help="""If the norm of the gradient vector exceeds this,
						normalize it to have the norm equal to max_grad_norm""")
	parser.add_argument("--drop_rate", type=float, default=0.5,
						help="Dropout rate.")

	# Model parameters
	parser.add_argument("--wp", type=int, default=10,
						help="Past context window size. Set wp to -1 to use all the past context.")
	parser.add_argument("--wf", type=int, default=10,
						help="Future context window size. Set wp to -1 to use all the future context.")
	parser.add_argument("--n_speakers", type=int, default=2,
						help="Number of speakers.")
	parser.add_argument("--hidden_size", type=int, default=100,
						help="Hidden size of two layer GCN.")
	parser.add_argument("--rnn1", type=str, default="lstm",
						choices=["lstm", "gru"], help="Type of RNN cell.")
	parser.add_argument("--rnn2", type=str, default="lstm",
						choices=["lstm", "gru"], help="Type of RNN cell.")					
	parser.add_argument("--class_weight", action="store_true",
						help="Use class weights in nll loss.")
	parser.add_argument('-kernel-num', type=int, default=100, help='number of each kind of kernel')
	parser.add_argument('-kernel-sizes', type=str, default='3,4,5', help='comma-separated kernel size to use for convolution')
	parser.add_argument('-static', action='store_true', default=False, help='fix the embedding')
	# others
	parser.add_argument("--seed", type=int, default=24,
						help="Random seed.")
	args = parser.parse_args()
	log.debug(args)
	args.kernel_sizes = [int(k) for k in args.kernel_sizes.split(',')]
	main(args)

