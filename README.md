#   AGRN: Accurate Gene Regulatory Network Inference using Collective Machine Learning Methods


### Table of Content

- [Setup](#getting-started)
- [Dataset](#Dataset)
- [Prerequisites](#Prerequisites)
- [Download and install code](#download-and-install-code)
- [Demo](#demo)
- [Run with Docker](#Run-with-Docker)
  
# Getting Started
 

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. 

 ## Dataset
The dataset can be found in the dataset directory. Both DREAM4 and DREAM5 datasets are inside the AGRN_tool.zip tool


## Prerequisites

You would need to install the following software before replicating this framework in your local or server machine.

 ```
Python version 3.7.4

Poetry version 1.1.12

You can install poetry by running the following command:

curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python -

To configure your current shell run `source $HOME/.poetry/env`


```
  
## Download and install code

- Retrieve the code

```
wget http://cs.uno.edu/~tamjid/Software/AGRN/AGRN_tool.zip
unzip AGRN_tool.zip

```

## Demo

To run the program, first, set the input path in the input.txt file. Here is a sample input file form DREAM5 E-coli Dataset.

```
./Dataset/DREAM5/training data/Network 1 - in silico/net1_expression_data.tsv
./Dataset/DREAM5/test data/DREAM5_NetworkInference_GoldStandard_Network1 - in silico.tsv
./Dataset/DREAM5/training data/Network 1 - in silico/net1_transcription_factors.tsv
./Dataset/DREAM5/training data/Network 1 - in silico/net1_gene_ids.tsv
```


Then, run following python command from the root directory.

```
cd AGRN_tool
poetry install
poetry run python AGRN.py

```

- Finally, check **output** folder for results. The output directory contains importance scores from ETR, SVR and RFR in csv files. The OutputResults.txt file shows the results in AUROC and AUPR.


## Run with Docker

- Build the docker image from Dockerfile.
```
export UID=$(id -u)
export GID=$(id -g)
docker build --build-arg USER=$USER \
             --build-arg UID=$UID \
             --build-arg GID=$GID \
             --build-arg PW=asdf \
             -t agrn\
             -f Dockerfile.txt\
             .
```

- Mount the Output direcotry in the Docker Container and run it.

```
docker run -ti  -v /$(pwd)/Output:/home/$USER/Output agrn:latest
```

- Then, run following python command from the root directory.
```
source $HOME/.poetry/env
poetry run python AGRN.py
```

- Finally, check **output** folder for results. The output should be available in both host and docker. The output directory contains importance scores from ETR, SVR and RFR in csv files along with a OutputResults.txt file that shows the results in AUROC and AUPR. 

## Authors

Duaa Mohammad Alawad, Md Wasi Ul Kabir, Md Tamjidul Hoque. For any issue please contact: Md Tamjidul Hoque, thoque@uno.edu 
