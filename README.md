# Method2comment 
Project on predicting natural language comments for Java methods using Neural Machine Translation and Graph Neural Networks.

## Training

To train the models run:

`python train.py trained_models single --patience 400 --run-name single`

The `single` dataset contains only one sample in the input. Larger datasets are not included in the repository but one can generate them by running (protos of the corpus need to be placed in a `corpus` directory).

 `python data_processing/generate_graph_dataset.py`.

One can adjust the hyperparameters in models/model_main.py. In particular to change the used architecture one should modify "encoder_type" setting it to one of "seq", "graph", "graph+seq", "seq+graph".

The repositository does not include the graph net library that needs to be placed in a `tf2_gnn` directory.

## Evaluation

To evaluate the trained and saved model run: 

`python evaluate.py trained_models/single_best_model.bin single`
