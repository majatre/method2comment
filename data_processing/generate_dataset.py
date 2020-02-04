from dataset import generate_dataset_from_dir
import pickle

# Generate data
methods_code, methods_comments = generate_dataset_from_dir(
    "../corpus-features")

# Store data
pickle.dump({'methods_code': methods_code, 'methods_comments': methods_comments}, 
    open('data/methods_code_comments_all.pkl', 'wb'))