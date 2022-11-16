# BERT4Rec on Tensorflow 2

## Get started
0. Install [Python](https://www.python.org/about/gettingstarted/) 
1. Clone this repository (`git clone url`)
2. Usage of pipenv as the virtual environment and dependency manager is recommended. 
Get [Pipenv](https://pypi.org/project/pipenv/)
3. Install dependencies via `pipenv install`. Add the `--dev` option afterwards to also install
development dependencies.
4. Optional: Configure GPU support for better performance. See this [section](./README.md#GPU-implementation)
5. Run application. The examples directory gives a brief overview on how to use this 
application.

## How to use
Have a look in the examples directory to get a quick overview on how to use this repository.
The application is build with OOP principles in mind. For most of the directories/modules 
you will find either a factory or a factory method to conveniently create concrete instances.
Most of them are also written with reasonable default values. The following code e.g. 
comfortably instantiates a SimpleTokenizer ready to use.
```python
from bert4rec import tokenizers

tokenizer = tokenizers.get()
```
For development purposes this project uses [pylint](https://pypi.org/project/pylint/) and 
[coverage](https://pypi.org/project/coverage/). These two python dependencies can also be run 
from the command line via `pipenv run` as they are included as pipenv scripts. Their behaviour
can be and is defined in their respective configuration files in the projects root directory
(`.coveragerc` and `.pylintrc`).

### Data preparation

```python
from bert4rec import dataloaders

# 1. Instantiate dataloader
# METHOD ONE: via abstract factory pattern
dataloader_factory = dataloaders.get_dataloader_factory()
dataloader = dataloader_factory.create_ml_1m_dataloader()
# custom values may be given via kwargs

# METHOD TWO: via direct class instantiation
dataloader_2 = dataloaders.BERT4RecML1MDataloader()

# The dataloader exposes many methods:
# generate vocab (basically fill the tokenizer vocab)
dataloader.generate_vocab()

# load the represented data into a dataset
ds = dataloader.load_data_into_ds()

# load the represented data directly into three separate datasets (train, validation, test)
train_ds, val_ds, test_ds = dataloader.load_data_into_split_ds()

# create a ranking of all the items in the dataset according to their popularity
pop_items_ranking = dataloader.create_popular_item_ranking()

# directly have the ranking tokenized
tokenized_pop_items_ranking = dataloader.create_popular_item_ranking_tokenized()

# directly generate three separate datasets already prepared for training, validation and
# testing tasks (items are tokenized and masked language model is applied
train_ds, val_ds, test_ds = dataloader.prepare_training()

# preprocess a dataset on your own
custom_train_ds = dataloader_2.preprocess_dataset(train_ds, apply_mlm = False)

# prepare input for inference tasks (e.g. recommendation on the basis of a "real" user sequence)
sequence = list(...)
model_input = dataloader.prepare_inference(sequence)
```

### Model instantiation

```python
import pathlib

from bert4rec import models
from bert4rec.models import model_utils
from bert4rec.models.components import networks
from bert4rec import utils

# METHOD ONE: via class instantiation and models parameters
vocab_size = 200
num_layers = 6
num_attention_heads = 8
# See class definition for which parameters can be set
encoder = networks.BertEncoder(vocab_size=vocab_size, 
                               num_layers=num_layers, 
                               num_attention_heads=num_attention_heads)
model_1 = models.BERTModel(encoder)


# METHOD TWO: via predefined models config (can be found and created in the respective configs directory)
# freely defined path
config_path = pathlib.Path("../config/bert_train_configs/ml-1m_128.json")
# or safer: path relative from project root
config_path = utils.get_project_root().joinpath("config/bert_train_configs/ml-1m_128.json")
encoder_config = utils.load_json_config(config_path)
encoder_2 = networks.BertEncoder(vocab_size, **encoder_config)
model_2 = models.BERTModel(encoder_2)


# METHOD THREE: via reloading of a saved models
# freely defined path
save_path = pathlib.Path("path/to/model_directory")
# or with utility functions 
save_path = model_utils.determine_model_path(pathlib.Path("model_directory_name"))
loaded_assets = models.BERT4RecModelWrapper.load(save_path)
loaded_wrapper = loaded_assets["model_wrapper"]
model_3 = loaded_wrapper.bert_model
dataloader_config = {}
if "tokenizer" in loaded_assets:
    tokenizer = loaded_assets["tokenizer"]
    dataloader_config["tokenizer"] = tokenizer
```

### Model training

```python
import pathlib
import tensorflow as tf

from bert4rec import models
from bert4rec.models.components import networks
from bert4rec import trainers
from bert4rec.trainers import optimizers

vocab_size = 200

# model is already instantiated/reloaded
encoder = networks.BertEncoder(vocab_size)
model = models.BERTModel(encoder)

# 1. Trainer instantiation
# METHOD ONE: via factory method
trainer = trainers.get(**{"model": model})
# METHOD TWO: via direct class instantiation
trainer_2 = trainers.BERT4RecTrainer(model)

# 2. initialize model for training
# METHOD ONE: use default values for model (as described in respective papers)
trainer.initialize_model()
# METHOD TWO: use custom values for model training
# 2.1 Optimizer:
# METHOD ONE: use default values with factory method
optimizer = optimizers.get()
# METHOD TWO: use custom values with factory method
optimizer_config = {
  "num_warmup_steps": 5000,
  "weight_decay_rate": 0.005,
  "epsilon": 1e-5,
}
optimizer_2 = optimizers.get(**optimizer_config)
# METHOD THREE: use function to create adamw optimizer (as the creation is a bit tricky)
optimizer_3 = optimizers.create_adam_w_optimizer(init_lr=6e-5, num_train_steps=200000, beta_1=0.8)

# 2.2 Loss
# only an example, the usage of the regular instead of the masked sparse categorical crossentropy
# is discouraged in conjunction with the masked language model
loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)

# 2.3 Metrics
# only an example, the usage of the regular instead of the masked sparse categorical accuracy
# is discouraged in conjunction with the masked language model
accuracy_metric = tf.keras.metrics.SparseCategoricalAccuracy()
...

metrics = [
  accuracy_metric,
  ...
]

trainer_2.initialize_model(optimizer_3, loss, metrics)

# 3. Train the model
train_ds = tf.data.Dataset()
val_ds = tf.data.Dataset()
# optional checkpoint path to save and possibly restore model weights from
checkpoint_path = pathlib.Path("path/to/checkpoints")
epochs = 10
history = trainer_2.train(train_ds, val_ds, checkpoint_path, epochs)
```

### Model evaluation
```python
import tensorflow as tf

from bert4rec import evaluation
from bert4rec import models
from bert4rec.models.components import networks

vocab_size = 200

# model is already instantiated/reloaded
encoder = networks.BertEncoder(vocab_size)
model = models.BERTModel(encoder)
model_wrapper = models.BERT4RecModelWrapper(model)

# 1. Evaluator instantiation
# METHOD ONE: via factory method with default values
evaluator = evaluation.get()
# METHOD TWO: via direct class instantiation with default values
evaluator_2 = evaluation.BERT4RecEvaluator()
# you may add custom values:
eval_metrics = [
  evaluation.Counter(name="Number Assessments"),
  evaluation.NDCG(10),
  evaluation.NDCG(20),
  evaluation.NDCG(50),
  evaluation.HR(1),
]
evaluator_3 = evaluation.get(**{"metrics": eval_metrics, "sample_popular": False})

# 2. Evaluate model
test_batches = tf.data.Dataset()
metrics_objects = evaluator_3.evaluate(model_wrapper, test_batches)
# when the sample_popular parameter is set to true (default) you have to give a 
# popular items ranking list or a dataloader which can generate this list

# The evaluate method returns the list of metric objects. If you want to have the metrics results
# in a dictionary you can call this method:
metric_results_dict = evaluator_3.get_metrics_results()
```


## Directory structure

- #### bert4rec
  - Contains the main application code
  - ##### apps
    - Tensorflow modules that implement a practical (well, at least theoretically) use case with 
"usable" output
  - ##### dataloaders
    - Classes to load data from dataset files in a consistent way and provide a standardized
for further usage. Each dataset is represented by a dataloader class
  - ##### evaluation
    - Code for evaluating the model. Evaluation is a little tricky as we often only have one
ground truth element and not a ground truth "sequence" to compare the model output with.
  - ##### model
    - The main directory for all the model and model components related code
  - ##### tokenizers
    - Contains tokenizers
  - ##### trainers
    - Contains training utilities and also the optimizer factory
  - ##### utils
    - Contains some generic utility functionality and "global" variables
- #### config
  - A directory for all types of configuration files necessary for the code
  - ##### bert_train_configs
    - BERT machine learning configurations for recommendation tasks according to this 
[paper](https://arxiv.org/abs/1904.06690) and this [repository](https://github.com/FeiSun/BERT4Rec)
- #### datasets
  - Python "helper" code to load and store raw datasets
- #### examples
  - Contains handy and brief examples about how to operate the codebase
- #### tests
  - Tests directory

### GPU implementation
*Note: Only works (officially) with Nvidia GPUs*

If you want to use GPU for improved performance, make sure to have the latest [drivers for your
GPU](https://www.nvidia.com/Download/index.aspx) installed. Then, follow this 
[guide](https://www.tensorflow.org/install/pip?hl=en).


### Further notes

Some parts of this repository (mainly the components directory and the components_tests directory)
have been copied from the [Tensorflow Model Garden](https://github.com/tensorflow/models) 
repository.\
Repository disclaimer:

> Copyright 2022 The TensorFlow Authors. All Rights Reserved.
> 
> Licensed under the Apache License, Version 2.0 (the "License");
> you may not use this file except in compliance with the License.
> You may obtain a copy of the License at
> 
> http://www.apache.org/licenses/LICENSE-2.0
> 
> Unless required by applicable law or agreed to in writing, software
> distributed under the License is distributed on an "AS IS" BASIS,
> WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
> See the License for the specific language governing permissions and
> limitations under the License.