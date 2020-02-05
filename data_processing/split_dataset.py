import pickle
import random

data = pickle.load(open('./data/methods_code_comments_all_without_jsoup.pkl', 'rb'))
methods_code = data['methods_code']
methods_comments = data['methods_comments']

data_pairs = list(zip(methods_code, methods_comments))
random.shuffle(data_pairs)

n = len(data_pairs)
subsets = {
    'train': data_pairs[:int(0.8*n)],
    'valid': data_pairs[int(0.8*n):int(0.9*n)],
    'test': data_pairs[int(0.9*n):]
}

for k, dataset in subsets.items():
    pickle.dump({'methods_code': [x[0] for x in dataset], 'methods_comments': [x[1] for x in dataset]}, 
        open('data/' + k + '.pkl', 'wb'))