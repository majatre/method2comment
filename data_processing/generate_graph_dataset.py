from dataset import generate_dataset_from_dir
import pickle
import jsonlines
import random
import gzip
import shutil


# Generate data
methods_code, methods_comments, graphs = generate_dataset_from_dir(
    "../corpus-features/libgdx")
name = "libgdx"

random.shuffle(graphs)

n = len(graphs)
subsets = {
    'train': graphs[:int(0.8*n)],
    'valid': graphs[int(0.8*n):int(0.9*n)],
    'test': graphs[int(0.9*n):]
}

for k, dataset in subsets.items():
    with jsonlines.open('data/'+k+'_'+name+'.jsonl', mode='w') as writer:
        writer.write_all(dataset)

for k in ['train', 'valid', 'test']:
    with open('data/'+k+'_'+name+'.jsonl', 'rb') as f_in:
        with gzip.open('jsonl_datasets/'+name+'/'+k+'.jsonl.gz', 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)