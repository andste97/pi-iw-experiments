# Policy-Based Heuristics for IW Experiments
Repository for my Thesis in Computer Science and Engineering at DTU.

Code is based on a mixture of the repositories for [Deep Policies for Width-Based Planning in Pixel Domains](https://arxiv.org/abs/1904.07091) and [Hierarchical Width-Based Planning and Learning](https://arxiv.org/abs/2101.06177), with my own code on top.

## Experiments
For atari games, use the deterministic version of the gym environments, which can be specified by selecting v4 environments (e.g. "Breakout-v4"). Although the "NoFrameskip" environment is given, we set the frameskip anyway with parameter ```--atari-frameskip``` (15 in our experiments).

## Installation
* Install the [requirements](requirements.txt)
* Make sure that the correct version of [gridenvs](https://github.com/andste97/gridenvs) are added to the python path.

### Installing Pygraphviz on Windows
Pygraphviz is necessary for generating the state tree images.

1. Install windows C++ 14.0 or newer build tools
2. Install graphviz from <https://gitlab.com/graphviz/graphviz/-/releases>
    * Choose that information will be added to PATH for user
3. Install pygraphviz from with command: `conda install -c alubbock pygraphviz` (will fail if not installed through conda)


## Running experiments

I have implemented several different version of the pi-IW algoritm: one deprecated version using pytorch-lightning 
and a newer one using pytorch vanilla (model_dynamic.py). The pytorch lightning versions have been deprecated,
because pytorch lightning was too inflexible for the experiments I needed to run and I had to give up on using it.

To train 3 models per environment, on multiple environments at once, use the following command:

`
python3 ../piiw/online_planning_learning_run_entire_suite.py
`

Be careful with how many parameters are in the env_suite config parameter, as too many entries in this list 
mean a lot of RAM and CPU is used. This command should also only be used on machines with a NVIDIA GPU.

To overwrite the list and train only on e.g. Pong, add the following command line param to the above command:

`'env_suite=["Pong-v4"]'`

## Docker containers

### Commands to build docker containers
1. Training container: `docker build -f dockerfiles/train_model.dockerfile . -t andi97/piiw_trainer:latest`

### Commands to run docker containers
The docker containers are set up without an entrypoint. 

#### Running the training container:
`docker run -v ./data:/data -v ./models:/models -e WANDB_API_KEY='<your-api-key>' andi97/piiw_trainer:latest python3 ./piiw/online_planning_learning_lightning.py`

IMPORTANT: to add GPU support to a container, add the flag `--gpus all` to above command, like so:

`docker run -v ./data:/data -v ./models:/models -e WANDB_API_KEY='<your-api-key>' --gpus all andi97/piiw_trainer:latest python3 ./piiw/online_planning_learning_lightning.py`

Other params:

We use hydra for our config. See their documentation for how to overwrite/add config values.
See: https://hydra.cc/docs/advanced/override_grammar/basic/

To overwrite the config to be used, use the key `--config-name <config_name.yaml>`

## Submitting jobs to HPC

To submit a job to the hpc, from the scripts directory, run command:

`bsub < <hpc-submission-script>`

More information can be found here:
* Login: <https://www.hpc.dtu.dk/?page_id=2501>
* Submission script: <https://www.hpc.dtu.dk/?page_id=1416>
* Batch jobs: <https://www.hpc.dtu.dk/?page_id=1416>
* 

## Sweeps

### Initializing a sweep
From project root run command: 
`wandb sweep --project <sweep-project-name> ./piiw/models/config/sweep.yaml`

### Starting sweep agents
From project root repository, run command:
`wandb agent piiw-thesis/piiw-sweep/<sweep-id>`


## Project structure

The directory structure of the project looks like this:

```txt

├── README.md            <- The top-level README for developers using this project.
├── dockerfiles               <- Dockerfiles to build images
│   ├── flaskapi.dockerfile
│   ├── inference_streamlit.dockerfile
│   ├── predict_model.dockerfile
│   └── train_model.dockerfile
│  
├── docs                 <- Documentation folder
│   │
│   ├── index.md         <- Homepage for your documentation
│   │
│   ├── mkdocs.yml       <- Configuration file for mkdocs
│   │
│   └── source/          <- Source directory for documentation files
│
├── pyproject.toml       <- Project configuration file
│
├── reports              <- Generated analysisbucketbucketbucketbucket as HTML, PDF, LaTeX, etc.
│   ├── README.md
│   │
│   ├── report.py
│   │
│   └── figures          <- Generated graphics and figures to be used in reporting
│
├── requirements.txt     <- The requirements file for reproducing the analysis environment
│
├── piiw  <- Source code for use in this project.
│   │
│   ├── __init__.py      <- Makes folder a Python module   
│   │
│   ├── atari_utils      <- custom atari wrappers and other code to make atari envs work with the code
│   │
│   ├── data             <- custom datasets and adapters to make pytorch lightning work with the experience replay dataset
│   │
│   ├── envs             <- custom gridenv levels
│   │
│   ├── models           <- model implementations, training script and prediction script
|   |   ├── config
|   |   |   |
|   |   |   ├── config_atari_dynamic.yaml <- the default configuration to be used for Atari. Configured for the final experiment shown in the thesis.
|   |   |   ├── config_basic.yaml <- the configuration to be used with lightning_model_basic.py. Only for gridenvs!
|   |   |   ├── config_dynamic.yaml <- the configuration to be used with lightning_model_dynamic.py. Only for gridenvs!
|   |   |   ├── sweep.yaml <- configuration used for wandb sweep with lightning model on Gridenvs
|   |   |   └── sweep_atari.yaml <- configuration used for wandb sweep with lightning model on Atari games
|   |   |
│   │   ├── __init__.py
|   |   ├── lightning_model_basic.py   <- contains the pytorch lightning version of pi-iw with BASIC features. Deprecated.
|   |   ├── lightning_model_dynamic.py    <- contains the pytorch lightning version of pi-iw with dynamic features. Deprecated.
|   |   ├── lightning_model_evaluator.py  <- contains a test script for running tests on checkpoints of the lightning models. deprecated.
|   |   ├── mnih_2013.py   <- contains the policy network definition
│   │   └── model_dynamic.py  <- latest implemented version of the PI-IW algorithm, with pytorch vanilla. Use this one for any experiments!
│   │
│   ├── novelty_tables    <- contains different novelty tables to use for width-based planners
│   │
│   ├── planners      <- contains different planners
│   │
│   ├── tree_utils      <- contains utility code to work with state trees
│   │
│   ├── utils      <- contains additional utility code not fitting anywhere else
│   │
│   ├── visualization    <- Scripts to create exploratory and results oriented visualizations
│   │   ├── __init__.py
│   │   └── visualize_tree_with_observations.py <- contains the tree visualizations script used for generating tree plots in the thesis.
│   │
│   ├── online_planning.py   <- run a full planning episode using basic features and the solver, using a uniform policy to sample actions
│   ├── online_planning_learning_lightning.py   <- script for training the model using lightning_model_basic.py or lightning_model_dynamic.py deprecated
│   ├── online_planning_learning_run_entire_suite.py   <- train multiple models in parallel in multiple environments. 3 models with different seeds/environment, multiple environments can be trained at once. Use this script for experiments.
│   ├── online_planning_learning_v2.py   <- train a single model in one environment using model_dynamic.py
└── └── planning_step.py   <- script for running a single planning step using basic features and the solver, using a uniform policy to sample actions  
```
