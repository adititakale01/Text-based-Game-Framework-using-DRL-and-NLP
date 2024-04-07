from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

fp = open("lstm_100_embeddings.txt", "r")
data = fp.read()
fp.close()
splitdata = data.split("\n")
length = len(splitdata) - 1
words = []
embedding = []
for i in range(length):
	if i % 2 == 0:
		words.append(splitdata[i])
	else:
		line = splitdata[i].split(' ')
		row = []
		for j in line:
			row.append(float(j))
		embedding.append(row)

model = TSNE(n_components=2, random_state=0, n_iter = 2000, perplexity = 50)
embeds = model.fit_transform(embedding)
# xmin = embedding[0][0]
# xmax = embedding[0][0]
# ymin = embedding[0][1]
# ymax = embedding[0][1]
# for i in range(len(embedding)):
# 	xmin = min(xmin, embedding[i][0])
# 	xmax = min(xmax, embedding[i][0])
# 	ymin = min(ymin, embedding[i][1])
# 	ymax = min(ymax, embedding[i][1])

# print xmin, xmax, ymin, ymax

x = [ele[0] for ele in embedding]
y = [ele[1] for ele in embedding]
plt.figure()
# plt.autoscale(enable=True, axis='both', tight=None)
# plt.xlim(xmin, xmax)
# plt.ylim(ymin, ymax)
plt.scatter(x, y, s = 0)
for i in range(len(words)):
	plt.annotate(words[i], xy = (x[i], y[i]))
plt.show()