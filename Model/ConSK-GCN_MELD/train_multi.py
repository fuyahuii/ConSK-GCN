
import argparse
import random

import torch
import pickle
import dgcn
import numpy as np
import os
from tqdm import tqdm
import time
import logging
import nni
from nni.utils import merge_parameter

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
	dgcn.utils.set_seed(args["seed"])
	
	# load data
	log.info("Loaded data.")
	with open('.\Text_MELD_Bert.pickle', 'rb') as handle:
		(train_text, train_label, dev_text, dev_label, test_text, test_label, maxlen, train_length, dev_length,
		 test_length) = pickle.load(handle)

	with open('.\Audio_MELD.pickle', 'rb') as handle:
		(train_audio, train_label, dev_audio, dev_label, test_audio, test_label, maxlen, train_length, dev_length,
		 test_length) = pickle.load(handle)

	with open('.\Knowledge_nrc_MELD.pickle','rb')as handle:
		(Concept_data_train,Concept_data_dev,Concept_data_test,Anew_data_train,Anew_data_dev,Anew_data_test)=pickle.load(handle)

	print(Concept_data_train.shape,Concept_data_test.shape)

	trainset = dgcn.Dataset(train_text,train_audio,train_label,train_length, Concept_data_train, Anew_data_train, args["batch_size"])
	devset = dgcn.Dataset(dev_text,dev_audio, dev_label, dev_length, Concept_data_dev, Anew_data_dev,args["batch_size"])
	testset = dgcn.Dataset(test_text,test_audio,test_label, test_length, Concept_data_test, Anew_data_test, args["batch_size"])

	log.debug("Building model...")
	model = dgcn.CONSKGCN(args).to(args["device"])
	opt = dgcn.Optim(args["learning_rate"], args["max_grad_value"], args["weight_decay"])
	opt.set_parameters(model.parameters(), args["optimizer"])

	coach = dgcn.Coach(trainset, devset,testset, model,opt,args)# Calculate f1 and loss

	# Train.
	log.info("Start training...")
	ret = coach.train()

	# Save.
	checkpoint = {
		"best_dev_f1": ret[0],
		"best_epoch": ret[1],
		"best_state": ret[2],
	}
	#torch.save(checkpoint, model_file) # entire net

def get_params():

	save_path="./"
	setup_logging(os.path.join(save_path, 'log_test_DialogueGCN_text_knowledge.txt'))
	
	parser = argparse.ArgumentParser(description="train.py")
	parser.add_argument("--data", type=str,default=False,
						help="Path to data")

	# Training parameters
	parser.add_argument("--from_begin", action="store_true", default=True,
						help="Training from begin.")
	parser.add_argument("--device", type=str, default="cuda",
						help="Computing device.")
	parser.add_argument("--epochs", default=15, type=int,
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
	parser.add_argument("--wp", type=int, default=6,
						help="Past context window size. Set wp to -1 to use all the past context.")
	parser.add_argument("--wf", type=int, default=6,
						help="Future context window size. Set wp to -1 to use all the future context.")
	parser.add_argument("--n_speakers", type=int, default=2,
						help="Number of speakers.")
	parser.add_argument("--hidden_size", type=int, default=100,
						help="Hidden size of two layer GCN.")
	parser.add_argument("--rnn1", type=str, default="lstm",
						choices=["lstm", "gru"], help="Type of RNN cell.")
	parser.add_argument("--rnn2", type=str, default="lstm",
						choices=["lstm", "gru"], help="Type of RNN cell.")
	parser.add_argument("--gcn_h1", type=float, default=100,
						help="dim of the gcn")
	parser.add_argument("--gcn_h2", type=float, default=100,
						help="dim of the gcn")
	parser.add_argument("--param_t", type=float, default=0.5,
						help="ratio between semantics and concept konwledges in text modality")
	parser.add_argument("--param_a", type=float, default=0.5,
						help="ratio between semantics and concept konwledges in audio modality")
	parser.add_argument("--class_weight", action="store_true",
						help="Use class weights in nll loss.")

	# others
	parser.add_argument("--seed", type=int, default=0,
						help="Random seed.")

	args, _ =parser.parse_known_args()
	return args

if __name__ == '__main__':

    try:
		# get parameters form tuner
        tuner_params = nni.get_next_parameter()
        log.debug(tuner_params)
        params = vars(merge_parameter(get_params(), tuner_params))
        print(params)
        main(params)
    except Exception as exception:
        log.exception(exception)
        raise
