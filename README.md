# AlaskaBurnedAreaModel

This repository contains the code, slurm scripts, outputs, paper and slide of my Master's project for USGS.

The code was executed on the HPC clusters provided by Research Computing Center at University of Colorado, Boulder and XSEDE.

This project utilized the Summit supercomputer, which is supported by the National Science Foundation (awards ACI-1532235 and ACI-1532236), the University of Colorado Boulder, and Colorado State University. All analyses were conducted on the SHAS partition, which consists of 380 nodes, 24 cores/nodes, 4.84 GB Ram/core

#### Programming Language: 
Python, slurm bash script

### Abstract:

Fires impact our ecosystems. In order to study the impact of past fires and forecast future patterns, a reliable and consistent record of fire is essential. However, very few agencies consistently track fire occurrence over space and time. The incompleteness of fire data makes assessing trends and impacts of fires challenging. To address this issue, USGS (United States Geological Survey) developed a machine learning algorithm to map burned areas for the Conterminous US using Landsat satellite images for date ranges from 1984 to present. The algorithm performs well in comparison to similar research; however, the error rates in USGS’s published research results in 2020 suggest rooms for potential improvement. Therefore, prior to expanding its fire mapping coverage from Conterminous US to the next biggest territory, Alaska, USGS sought to modify current algorithm for better prediction performance. This project is inspired by, and partially belongs to USGS’s burned area mapping expansion effort, which aims to develop accurate and efficient tools for mapping burned areas in Alaska. In particular, in this project, 5 aspects of the existing USGS algorithm: data-split, model evaluation metric, classifier, feature selection, hyperparameter tuning were examined and discussed. Using the existing USGS algorithm as a baseline model, the 5 aspects were modified one by one progressively for experiments and result benchmarking. At each stage of the modification process, performance accuracy and efficiency were compared between the models before and after the modification. Overall, at the end of the modification process, an improvement of approximately 5.5% in Average-Precision score was achieved. Omission Error and Commission Error were reduced by 9.74% and 2.28% respectively. The process of model training, tuning and evaluation was shortened from a total of 3 Days 18 hours (90 hours) to 19 hours utilizing fewer computation and parallelization resources.
