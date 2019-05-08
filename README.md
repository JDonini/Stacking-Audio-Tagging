Stacking Audio Tagging
======================

In this work I proposed apply Stacking with Convolutional Neural Network to improve Audio Tag classification.

Project Organization
--------------------
```

├── config-project.yml
├── database
│   └── CAL500
│       ├── config.py
│       ├── data
│       │   ├── processed
│       │   │   ├── annotations
│       │   │   │   ├── train.csv
│       │   │   │   ├── test.csv
│       │   │   │   └─ validation.csv
│       ├── __init__.py
│       ├── models
│       ├── out
│       └── src
│           ├── cnn-cnn-complex
│           │   ├── first_stage.py
│           │   ├── __init__.py
│           │   ├── model.py
│           │   └── second_stage.py
│           ├── cnn-cnn-simple
│           │   ├── first_stage.py
│           │   ├── __init__.py
│           │   ├── model.py
│           │   └── second_stage.py
│           ├── cnn-svm
│           │   ├── first_stage.py
│           │   ├── __init__.py
│           │   ├── model.py
│           │   └── second_stage.py
│           └── cnn-svm-svm
│               ├── first_stage.py
│               ├── __init__.py
│               ├── model.py
│               └── second_stage.py
├── docs
│   ├── conf.py
│   ├── index.rst
│   └── Makefile
├── environment
│   ├── environment.yml
│   ├── install_environment.sh
│   ├── update_packages_environment.sh
│   └── update_yml_environment.sh
├── LICENSE
├── main.py
├── README.md
├── scripts
│   └── get_database.sh
└── src
    ├── __init__.py
    ├── check_data.py
    ├── generate_graph.py
    ├── generate_holdout.py
    ├── generate_spectrogram.py
    ├── generate_structure.py
    ├── metrics.py
    └── preprocessing_data.py
```
--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
