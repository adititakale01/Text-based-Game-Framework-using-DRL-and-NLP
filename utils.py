import sys
import string
import pprint

try:
    xrange
except NameError:
    xrange = range

pp = pprint.PrettyPrinter()

def clean_words(words):
  return [word.lower().translate(None, string.punctuation) for word in self.idx2word]

def load_data(fileName):
	with open(fileName, "r") as fp:
		data = fp.read()
		splitdata = data.split('\n')
		state = []
		QActionValues = []
		QObjectValues = []
		i = 0
		while i < (len(splitdata) - 1):
			state.append([int(element) for element in splitdata[i].split(' ')])
			i += 1
			QActionValues.append([float(element) for element in splitdata[i].split(' ')])
			i += 1
			QObjectValues.append([float(element) for element in splitdata[i].split(' ')])
			i += 1
	return state, QActionValues, QObjectValues