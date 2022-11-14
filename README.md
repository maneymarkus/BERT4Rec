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