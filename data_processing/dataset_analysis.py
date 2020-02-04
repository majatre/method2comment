import pickle
import matplotlib.pyplot as plt
import numpy as np


data = pickle.load(open('./data/methods_code_comments_all.pkl', 'rb'))
methods_code = data['methods_code']
methods_comments = data['methods_comments']

print("The corpus contains {} methods with comments".format(len(methods_comments)))

methods_length = [len(method) for method in methods_code]
print("The average method length is: {}".format(np.mean(methods_length)))
plt.xlabel('No. tokens in method')
plt.ylabel('Frequency')
a = plt.hist(methods_length, bins=100, range=(0, 500), )
plt.show()

methods_comments_length = [len(method_comment.split()) for method_comment in methods_comments]
print("The average number of comment length is: {}".format(np.mean(methods_comments_length)))
plt.xlabel('No. words in method comment')
plt.ylabel('Frequency')
a = plt.hist(methods_comments_length, bins=range(0, 200))

plt.show()