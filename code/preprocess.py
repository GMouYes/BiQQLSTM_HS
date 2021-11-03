#Input: list of sentence
#Desired output: 2 list of sentences
import random
import numpy as np
from transformers import BertTokenizer, BertModel
import time
import torch

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


def getDict():
	with open('adjectives_people.txt') as f:
		content = f.readlines()
	return [' '+x.strip()+' ' for x in content]

def embedding(data:list):
	inputs = tokenizer(data, padding='max_length', truncation=True, max_length=32, return_tensors="pt")
		
	return inputs

def overlapPharse(sentence:str, pharse:list):
	overlap = []
	# sentence = ' '+sentence+' '
	overlap = [word for word in pharse if word in sentence ]
	return set(overlap)


def perturb(data:list, test=False):
	data = list(data)
	if test:
		dataEmb = embedding(data)
		return dataEmb, dataEmb

	adjective_people = getDict()
	perturbData = []
	for sentence in data:
		tmpSentence = ' '+sentence +' '
		overlap = overlapPharse(tmpSentence, adjective_people)
		
		if len(overlap)>=1:
			targetWord = random.sample(overlap,1)[0]
			destinationWord = random.sample([x for x in adjective_people if x != targetWord],1)[0]
			replacedSentence = tmpSentence.replace(targetWord, destinationWord)
			perturbData.append(replacedSentence.strip())

		else:
			perturbData.append(sentence)

	# length = len(data)
	dataInputs = embedding(data)
	perturbInputs = embedding(perturbData)

	return dataInputs, perturbInputs
