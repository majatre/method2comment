import pickle
import matplotlib.pyplot as plt
import numpy as np


data = pickle.load(open('./data/valid.pkl', 'rb'))
methods_code = data['methods_code']
methods_comments = data['methods_comments']

print("The corpus contains {} methods with comments".format(len(methods_comments)))

methods_length = [len(method) for method in methods_code]
print("The average method length is: {:.2f}".format(np.mean(methods_length)))
print("The {:.2f} of methods are shorter than 200 tokens".format(
    len(list(filter(lambda m: m < 200, methods_length)))
    /len(methods_length)*100))

plt.xlabel('No. tokens in method')
plt.ylabel('Frequency')
a = plt.hist(methods_length, bins=100, range=(0, 500), )
# plt.show()

methods_comments_length = [len(method_comment.split()) for method_comment in methods_comments]
print("The average number of comment length is: {:.2f}".format(np.mean(methods_comments_length)))
print("The {:.2f} of comments are shorter than 50 tokens".format(
    len(list(filter(lambda c: c < 50, methods_comments_length)))
    /len(methods_comments_length)*100))
plt.xlabel('No. words in method comment')
plt.ylabel('Frequency')
a = plt.hist(methods_comments_length, bins=range(0, 200))
# plt.show()