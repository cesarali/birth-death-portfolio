# __Discrete States Generative Diffusion Model Library__

## __Abstract__

Framework for __training __ that implements generative diffusion models for discrete data.

## __Framework Structure__

For unified training and evaluation and easy implementation of 
new diffusion models we define in our framework two basic concepts,
_Model_ and _Model Trainer_.

__Model__. This concept is implemented as abstract class ``AModel`` in [discrete-diffusion.models]. The UML diagram of the class can be find below.

![AModel](reports/figures/AModel.png)

Every dynamic language method implemented in this framework must be a child class from [discrete-diffusion.models.AModel]. In order one to use all feature (training, logging, ...) of the `supervisedllm` framework one must implement the new models as sub-class of ``AModel`` class. This abstract class defines four abstract methods that are used for the training and evaluation that needed to be implemented:

  1. ``forward()``: this methods implements the inference of the model on a data point,
  2. ``train_step()``: the procedure executed during one training step
  3. ``validate_step``: the procedure executed during one validation step
  4. ``loss()``: definition of the loss for each specific model that will be used for training.

Example implementation of the Faster R-CNN model can be found in [discrete-diffusion.models.baseline_models.SequentialClassifier]

__Model Trainer__.  This class is used for training models implemented in this framework. 
The class is handeling the model trainin, logging and booking. In order this class to be used the models must be child class from the [supervisedllm.models.AModel]. This means that the model must implement four abstract functions from the parent class (see above).

## __Installation__

In order to set up the necessary environment:

### __Virtualenv__

1. Install [virtualenv] and [virtualenvwrapper].
2. Create virtualenviroment for the project:

    ```bash
    mkvirtualenv discrete_diffusion
    ```

3. Install the project in edit mode:

    ```bash
    python setup.py develop
    pip install -e .
    ```

Optional and needed only once after `git clone`:

1. install several [pre-commit] git hooks with:

   ```bash
   pre-commit install
   # You might also want to run `pre-commit autoupdate`
   ```

   and checkout the configuration under `.pre-commit-config.yaml`.
   The `-n, --no-verify` flag of `git commit` can be used to deactivate pre-commit hooks temporarily.

2. install [nbstripout] git hooks to remove the output cells of committed notebooks with:

   ```bash
   nbstripout --install --attributes notebooks/.gitattributes
   ```

   This is useful to avoid large diffs due to plots in your notebooks.
   A simple `nbstripout --uninstall` will revert these changes.

Then take a look into the `scripts` and `notebooks` folders.

## __Project Organization__

```
├── AUTHORS.md              <- List of developers and maintainers.
├── CHANGELOG.md            <- Changelog to keep track of new features and fixes.
├── CONTRIBUTING.md         <- Guidelines for contributing to this project.
├── Dockerfile              <- Build a docker container with `docker build .`.
├── LICENSE.txt             <- License as chosen on the command-line.
├── README.md               <- The top-level README for developers.
├── configs                 <- Directory for configurations of model & application.
├── data
│   ├── external            <- Data from third party sources.
│   ├── interim             <- Intermediate data that has been transformed.
│   ├── processed           <- The final, canonical data sets for modeling.
│   └── raw                 <- The original, immutable data dump.
├── docs                    <- Directory for Sphinx documentation in rst or md.
├── requrements.txt         <- The python environment file for reproducibility.
├── models                  <- Trained and serialized models, model predictions,
│                              or model summaries.
├── notebooks               <- Jupyter notebooks. Naming convention is a number (for
│                              ordering), the creator's initials and a description,
│                              e.g. `1.0-fw-initial-data-exploration`.
├── pyproject.toml          <- Build configuration. Don't change! Use `pip install -e .`
│                              to install for development or to build `tox -e build`.
├── references              <- Data dictionaries, manuals, and all other materials.
├── reports                 <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures             <- Generated plots and figures for reports.
├── scripts                 <- Analysis and production scripts which import the
│                              actual PYTHON_PKG, e.g. train_model.
├── setup.py                <- Use `python setup.py develop` to install for
│                              development or `python setup.py bdist_wheel` to build.
├── src
│   └── kiwissenbase        <- Actual Python package where the main functionality goes.
├── tests                   <- Unit tests which can be run with `pytest`.
├── .coveragerc             <- Configuration for coverage reports of unit tests.
├── .isort.cfg              <- Configuration for git hook that sorts imports.
└── .pre-commit-config.yaml <- Configuration of pre-commit git hooks.
```
