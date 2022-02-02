import os
import time
import argparse
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data

import model 
from metrics import metrics
from dataset import MovieLensDataset



if __name__ == 'main':
	parser = argparse.ArgumentParser()
	parser.add_argument("--DATA_PATH",
		type=str,
		default=None,
		help="DATA_PATH")
	parser.add_argument("--zipfile",
		type=bool,
		default=False,
		help="zipfile")
	parser.add_argument("--seed", 
		type=int, 
		default=42, 
		help="Seed")
	parser.add_argument("--lr", 
		type=float, 
		default=0.001, 
		help="learning rate")
	parser.add_argument("--dropout", 
		type=float,
		default=0.2,  
		help="dropout rate")
	parser.add_argument("--batch_size", 
		type=int, 
		default=256, 
		help="batch size for training")
	parser.add_argument("--epochs", 
		type=int,
		default=30,  
		help="training epoches")
	parser.add_argument("--top_k", 
		type=int, 
		default=10, 
		help="compute metrics@top_k")
	parser.add_argument("--embedding_dim", 
		type=int,
		default=64, 
		help="predictive factors numbers in the model")
	parser.add_argument("--mlp_layers",
	    nargs='+', 
	    default=[128,64,32,16],
	    help="MLP layers. Note that the first layer is the concatenation of user and item embeddings. So layers[0]/2 is the embedding size.")
	parser.add_argument("--n_negs", 
		type=int,
		default=5, 
		help="Number of negative samples for training set")
	parser.add_argument("--num_ng_test", 
		type=int,
		default=100, 
		help="Number of negative samples for test set")

	args = parser.parse_args()
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	# DATA_PATH = DATA_PATH
	DATA_PATH = './data/ratings.csv'
	ml = MovieLensDataset(DATA_PATH, zipfile=args.zipfile)
	train_loader, test_loader = ml.data_loader(batch_size=args.batch_size)

	n_usrs, n_items = len(ml.user_pool) + 1, len(ml.item_pool) + 1
	model = model.NeuMF(n_users, n_items, args.embedding_dim, args.mlp_layers)
	model = model.to(device)

	loss_function = nn.BCELoss()
	optimizer = optim.Adam(model.parameters(), lr=args.lr)

	best_hr = 0
	for epoch in range(1, args.epochs + 1):
		model.train()
		start_time = time.time()

		for user, item, label in train_loader:
			user = user.to(device)
			item = item.to(device)
			label = label.to(device)

			optimizer.zero_grad()
			pred = model(user, item)
			loss = loss_function(pred, label)
			loss.backward()
			optimizer.step()

		model.eval()
		HR, NDCG = metrics(model, test_loader, args.top_k, device)

		elapsed_time = time.time() - start_time
		print("The time elapse of epoch {:03d}".format(epoch) + " is: " + 
				time.strftime("%H: %M: %S", time.gmtime(elapsed_time)))
		print("HR: {:.3f}\tNDCG: {:.3f}".format(np.mean(HR), np.mean(NDCG)))

		if HR > best_hr:
			best_hr, best_ndcg, best_epoch = HR, NDCG, epoch

	print("End. Best epoch {:03d}: HR = {:.3f}, NDCG = {:.3f}".format(
										best_epoch, best_hr, best_ndcg))


