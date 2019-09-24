Stacking Audio Tagging
======================

In this work I proposed apply Stacking with Convolutional Neural Network to improve Audio Tag Classification.

The project is organized into 3 main projects:
--------------------
    M1 - Utilize Stacking technique with CNN to improve Audio Tag Classification;
    M2 - Utilize Stacking technique with CNN and late fusion to improve Audio Tag Classification;
    M3 - Utilize Stacking technique with CNN and Autoencoders to improve Audio Tag Classification;

Project Organization
--------------------
```

├── config-project.yml
├── database
│   └── CAL500
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
│           ├── model-1
│           │   ├── first_stage.py
│           │   ├── __init__.py
│           │   ├── model.py
│           │   └── second_stage.py
│           ├── model-2
│           │   ├── first_stage.py
│           │   ├── __init__.py
│           │   ├── model.py
│           │   └── second_stage.py
│           ├── model-3
│           │   ├── first_stage.py
│           │   ├── __init__.py
│           │   ├── model.py
│           │   └── second_stage.py
│           └── model-n
│               ├── first_stage.py
│               ├── __init__.py
│               ├── model.py
│               └── second_stage.py
├── environment
│   ├── environment.yml
│   ├── install_environment.sh
│   ├── update_packages_environment.sh
│   └── update_yml_environment.sh
├── LICENSE
├── main.sh
├── README.md
├── scripts
│   └── get_database.sh
└── src
    ├── __init__.py
    ├── check_data.py
    ├── generate_autoencoders_chromagram.py
    ├── generate_autoencoders_mel_spectrogram.py
    ├── generate_autoencoders_mfcc.py
    ├── generate_autoencoders_stft.py
    ├── generate_graph.py
    ├── generate_holdout.py
    ├── generate_info_data.py
    ├── generate_spectrogram.py
    ├── generate_structure.py
    ├── metrics.py
    ├── model_autoencoders_chromagram.py
    ├── model_autoencoders_mel_spectrogram.py
    ├── model_autoencoders_mfcc.py
    └── model_autoencoders_stft.py

```
--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
