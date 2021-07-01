# Validation on Real Data of an Extended Embryo-Uterine Probabilistic Graphical Model for Embryo Selection

This repository contains all the code used in the development of a Master's thesis submitted as part of the MSc in Fundamental Principles of Data Science supervised by Jerónimo Hernández-González and Jesús Cerquides. The project deals with the problem of embryo selection, using a probabilistic graphical model to select the most promising ones. An extensive experimental setup is assembled to validate the model using data from real patients following ART treatments. The [Master's Thesis report](Master_Thesis_Report) can also be found in this repository.

## PGM for IVF
Embryo selection is a critical step in assisted reproduction (ART): a good selection criteria is expected to increase the probability of inducing pregnancy. In the past, machine learning methods have been used to predict implantation and to rank the most promising embryos. One of the main obstacles when dealing with this type of data is that it is only partially labeled. Current tecniques are able to determine the number of embryos implanted in a cycle but not their identity.

In this project we use a probabilistic graphical model that is able to learn even in the presence of latent variables. We use an Expectationn-Maximization (EM) algorithm to learn in this scenario. The model assumes independence between cycles and embryos and accounts for a third source of uncertainty corresponding to unknown factors.

<p align="center"><img src="https://user-images.githubusercontent.com/72464030/124105062-24443780-da63-11eb-9e99-5f9f80b177ef.PNG"  align=middle width=600pt />
</p>
<p align="center">
<em>Probabilistic graphical model. Shadowed nodes represent observed variables. Double line denotes a deterministic variable.</em>
</p>

The model is compared against a set of simpler baseline methods to test different properties and assumptions of the model. In particular we defined the following methods:

* **Baseline_0**: Assumes that all embryos with unknown outcome are negative. Only uses embryo features.
* **Baseline_cycles**: Assumes that all embryos with unknown outcome are negative. Uses both embryo and cycle features.
* **Naive EM**: Uses an EM learning algorithm that deals with all unknown outcome embryos equally. That is, without providing any additional information. Only uses embryo features.
* **EM with label proportions**: Uses an EM learning algorithm where label proportions of partially implanted cycles are provided. Only uses embryo features.

## Structure of the repository
+ All functions, classes and the learning algorithms are stored in [src](src).
+ The training of the models take place in their respective experiments code (e.g., [baseline_0.py](baseline_0.py)). Running these files starts the whole process: loading the data and preprocessing it (directly in the initialization method of the dataset class), creating the EM object (if needed) and training the corresponding model. Results are automatically stored in pickle files in the [results](results) folder. The different experiments are:
  * [Probabilistic graphical model](fullmodel.py)
  * [Probabilistic graphical model (hiding the ASEBIR score)](fullmodel_hidden.py)
  * [Baseline_0](baseline_0.py)
  * [Baseline_cycles](baseline_cycles.py)
  * [Naive EM](baseline_NaiveEM.py)
  * [EM with LP](baseline_EM.py)
+ Notebook [data_exploration.ipynb](data_exploration.ipynb) presents an exploration of the data, investigating some useful initial insights.
+ Notebook [results.ipynb](results.ipynb) shows the results for the probabilistic graphical model along with all the baseline methods to compare against.
+ The [Master's Thesis report](Master_Thesis_Report).

## Data
Since the data describes real patients following ART treatments we are not allowed to publish it here. If you want to obtain the data in order to reproduce the results you can request it writing to [Adrián Torres](mailto:adriantorresmartin@gmail.com) and your petition will be evaluated.

Once you receive the datasets you only have to include them in the empty [data](data) folder.
