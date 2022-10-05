# ML2018 - Project 1 - epflml higgs

## Structure

```
.
├── data                                // Location of input data
│   ├── data-description.md
│   ├── sample-submission.csv
│   ├── test.csv
│   └── train.csv
├── environment.yml                     // Use this to set up a corresponding conda environment
├── report.pdf
├── README.md                           // You are here!
├── src
│   ├── data-analysis                   // Preliminary data analysis & grid search analysis
│   │   ├── grid_search_analysis.ipynb
│   │   ├── grid_search_analysis.py
│   │   ├── data-analysis.ipynb
│   │   └── data-analysis.py
│   ├── functions                       // ML functions and "raw" classes
│   │   ├── helpers.py
│   │   └── implementations.py
│   ├── main.ipynb                      // Main model creation file
│   ├── main.py
│   ├── models                          // Models
│   │   ├── LinearClassifier.py
│   │   └── MultiClassifierModel.py
│   ├── preprocessing.py                // Preprossing
│   ├── run.py                          // Submission run file
│   └── utils                           // General utility functions
│       ├── csv.py
│       ├── grid_search.py
│       ├── jupyter.py
│       ├── logs.py
│       ├── misc.py
│       └── plots.py
└── submissions                         // Location of output data
    ├── grid_search_results.json
    └── final_prediction.csv
```


The notebook files were converted using this [script](https://gist.github.com/samuelsmal/144e1204d646cd65ff8864d4b483f948), but should be viewed as a notebook.

### Machine Learning Model implementations

Can be found under `./src/functions/implementations.py`.

## Submission execution

```
# this will train a model and save the predictions in `submissions/final_prediction.csv`
python run.py
```

## Report

## Authors (in alphabetical order)

Francesco Bardi, Samuel Edler von Baussnern, Zeya Yin
