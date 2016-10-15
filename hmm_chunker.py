#! /usr/bin/python
from __future__ import division
from nltk.corpus import brown
import numpy
import sys
from nltk.corpus import conll2000, conll2002

CHUNK_SET = set()
CHUNK_LIST = []
TAG_SET = set()
TAG_LIST = []
V = set()

transition = {}
emission_1 = {}
emission_2 = {}
context = {}

def printdic(dicti):
	for key,value in dicti:
		print key,value

def laplace(ngram, dictionary, type):
	k = 0.5
	wn_0 = ngram[0]
	wn_1 = ngram[1]	
	bgram = (wn_0, wn_1)		
	if type == 0:
		total = len(TAG_SET)		
	elif type == 1:
		total = len(V)
	else:
		total = len(CHUNK_SET)
	prob = 1.0*(dictionary.get(bgram, 0) + k)/(context.get(wn_0, 0) + k*total)
	return prob



def viterbi(sentence):
	best_score = {}
	best_edge = {}
	best_score["0:<sen_CH_BEG>"]=0
	best_edge["0:<sen_CH_BEG>"]=None
	
	i=0
	for i in range(len(sentence)):
		for prev_chunk in CHUNK_LIST:			
			for cur_chunk in CHUNK_LIST:								
				if(best_score.get(str(i)+":"+prev_chunk,-1)!=-1 and transition.get((prev_chunk,cur_chunk),-1)!=-1):					
					chunk_gram = (prev_chunk, cur_chunk)
					trans = laplace(chunk_gram,transition, 2)
					word_gram = (cur_chunk,sentence[i][0])
					tag_gram = (cur_chunk,sentence[i][1])
					emit1 = laplace(word_gram,emission_1, 1)		
					emit2 = laplace(tag_gram,emission_2, 1)		
					score = best_score[str(i)+":"+prev_chunk]+(-1*numpy.log10(trans))+(-1*numpy.log10(emit1))+(-1*numpy.log10(emit2))										
					if(best_score.get(str(i+1)+":"+cur_chunk,-1)==-1 or  best_score.get(str(i+1)+":"+cur_chunk,-1)>score):
						best_score[str(i+1)+":"+cur_chunk]=score
						best_edge[str(i+1)+":"+cur_chunk]=str(i)+":"+prev_chunk					

	cur_chunk="<sen_CH_END>"		
	for prev_chunk in CHUNK_LIST:					
		if(best_score.get(str(i)+":"+prev_chunk,-1)!=-1 and transition.get((prev_chunk,cur_chunk),-1)!=-1):					
			chunk_gram = (prev_chunk, cur_chunk)			
			trans = laplace(chunk_gram,transition, 2)			
			score = best_score[str(i)+":"+prev_chunk]+(-1*numpy.log10(trans))			
			if(best_score.get(str(i+1)+":"+cur_chunk,-1)==-1 or  best_score.get(str(i+1)+":"+cur_chunk,-1)>score):				
				best_score[str(i+1)+":"+cur_chunk]=score
				best_edge[str(i+1)+":"+cur_chunk]=str(i)+":"+prev_chunk								
				
	chunks = []
	chunks.append("<sen_CH_END>")
	i=len(sentence)-1
	next_edge = best_edge[str(i)+":<sen_CH_END>"]
	
	while(next_edge!="0:<sen_CH_BEG>"):
		position= next_edge.split(":")[0]
		chunk= next_edge.split(":")[1]				
		chunks.append(chunk)
		next_edge=best_edge[next_edge]
	
	chunks.reverse()	
	return chunks	
	
def preprocess(sentences):
	global V, transition, context, emission_1 ,emission_2
	sent_start_chunk='<sen_CH_BEG>'
	sent_end_chunk='<sen_CH_END>'
	for sentence in sentences:		
		for threepair in sentence:
			V.add(threepair[0])
			TAG_SET.add(threepair[1])
			CHUNK_SET.add(threepair[2])			
			context[threepair[2]] = context.get(threepair[2], 0) + 1
		CHUNK_SET.add(sent_start_chunk)
		CHUNK_SET.add(sent_end_chunk)
		context[sent_start_chunk] = context.get(sent_start_chunk, 0) + 1
		context[sent_end_chunk] = context.get(sent_end_chunk, 0) + 1
			

	for sentence in sentences:
		chunk_gram = ('<sen_CH_BEG>', sentence[0][2])		
		transition[chunk_gram] = transition.get(chunk_gram, 0) + 1				
		i = 1		
		while i < len(sentence):
			chunk_gram = (sentence[i-1][2], sentence[i][2])		
			transition[chunk_gram] = transition.get(chunk_gram, 0) + 1
			i += 1		
		chunk_gram = (sentence[i-1][2],'<sen_CH_END>')		
		transition[chunk_gram] = transition.get(chunk_gram, 0) + 1				
	
	#emmision_1 -> chunk,word 
	#emmision_2 -> chunk,tag 
	for sentence in sentences:
		for word in sentence:
			word_gram = (word[2], word[0])
			emission_1[word_gram] = emission_1.get(word_gram, 0) + 1
			word_gram = (word[2], word[1])
			emission_2[word_gram] = emission_2.get(word_gram, 0) + 1
	
	pass

if __name__ == "__main__":	
	
# TRAINING THE MODEL	
	data = conll2000.iob_sents()[:5000]
	preprocess(data)	
	
	
# TESTING THE MODEL
	TAG_LIST = list(TAG_SET)
	CHUNK_LIST = list(CHUNK_SET)
	test_sentences = conll2000.iob_sents()[5001:5050]			
	finalaccuracy=0.0	
	for sent in test_sentences:
		token_accuracy = 0					
		result = viterbi(sent)
		for w,x,y in sent:
			print w,
		print 
		i=0
		phrase=[]
		print ' '.join(result)
		print 'Extracted phrases:'
		for i in range(len(sent)-1):
			if(result[i]=='B-NP' and result[i+1]=='I-NP'):
				phrase.append(sent[i][0])				
				for j in range(i+1,len(sent)-2):
					if(result[j]!='I-NP'):
						print ' '.join(phrase)
						phrase=[]
						i=j
						break
					else:
						phrase.append(sent[j][0])		
		match =0 
		total = len(sent)
		i=0
		for i in range(len(sent)-1):			
			if (sent[i][2]==result[i]):
			 	match=match+1
		finalaccuracy+=float((match/total)*100)
		print 'sentence accuracy',(match/total)*100
		print ('\n')

	print ('\n')
	print ('\n')
	print "Final accuracy:"+str(float(finalaccuracy/len(test_sentences)))