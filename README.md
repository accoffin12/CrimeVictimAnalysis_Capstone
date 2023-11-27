# Utilizing K-Means, DBSCAN and Hierarchy Analysis to Improve Crime Prevention Through the Examination of Victimology Data
## A. C. Coffin 
### October 2023
---

## Introduction:
This project was created to demonstrated the exploration into the developement of machine learning models capable of predicting groups of women at risk geographically and provide an increased awareness of crime in America. The three parameters being examined in this project are age, crime type, and geographical location when discussing female victimology. This project was developed as a way to detect patterns among female victims both on a social level through the use of KMeans Clustering and on a geographical model using DBSCAN. Crime among female victims is complex, but understanding the general relationship between these factors and the crimes that femal victims experience can aid in imporivng crime prevention stratages both regionally and socially. Two clustering models were utilized to demonstrate the application of these models to victim data as well as examine the relationship between these three aspects on both a National and City level. 

## Table of Contents:
- [Prerequisites](#Prerequisites)
- [Instalation](#Instalation)
- [Data_Sources](#Data_Sources)
- [Files](#Files)
    -[Raw Data](#raw-data)
    -[Data](#Data)
    -[ML_PreProcess](#ml_preprocess)
- [Deployment](#Deployment)
- [Challenges](#Challenges)
- [Results](#Results)
- [Credits](#Credits)
- [References](#references)

## Prerequisites:
1. Git
2. Python 3.7 (3.11+ preferred)
3. Jupyter Notebooks
4. VS Code Editor
5. VS Code Extesion: Python (by Microsoft)

## Installation:
The following instructions allow for the instalation of packages and creation of an enviroment to run this notebook.

The following moduels are required:
```
import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.tools as tls
import plotly.express as px
import statsmodels.stats.outliers_influence 
import statsmodels.tools.tools 
import statistics as stat
import seaborn as sns
import seaborn.objects as so
import matplotlib.dates as mdates
import datetime as dt
import os
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics import calinski_harabasz_score
from sklearn.metrics import davies_bouldin_score
from sklearn.preprocessing import StandardScaler
from yellowbrick.cluster import SilhouetteVisualizer
```
1. Clone the repository to your local machine and navigate the project direvtory. Use the following command to create an enviroment, when promted in VS Code to set the .env to a workspace folder, select yes.

```
python -m venv .venv
```
2. Once created, activate the enviroment using the following call:
```
.venv\Scripts\activate
```
3. Check that all packages are installed, this includes yellow brick. 
4. Refer to README.md for additional information and recomented tutorials.
5. Use the following steps to begin coding. 
    a. Aquire the data source
    b. Crime Victim Data Exploration Analysis
    c. KMeans Clustering Models Developed for Predictive Modeling
        - Import the Data from ML_PreProcess
        - Scaling the Data
        - Designing the KMeans Model
        - Creating visualizations
        - Model Evaluation
    d. NYPD DBSCAN Clustering
        - Import the Data from ML_PreProcess 
        - Determining Data Characteristics
        - Building the Model
        - Visualizations & Evaluating the Model
    e. Discussion
    f. Limitations & Future Works
    g. Conclusion

Note: When running the yellowbrick portions of this notebook, coment out the imports for matplotlib and seaborn. These two will interfear with Yellowbrick when run and result in all of the plots using these two libraries to not function correctly.

## Data_Sources:
This project uses a combination of sources, each of them has been linked bellow along with descriptions of the source. All of the data had inidially been pulled into an SQL database and then was exported as CSV files. Each CSV has been listed in the files list.

* [Bureau of Justice Statistics, N-Dash](https://ncvs.bjs.ojp.gov/multi-year-trends/crimeType): The official page of the BJS for the NCVS Dashboard. Each of these files takes the form of an independent quiery. Please note that these will produce independed CSV files and require processing for a full data set.
* [NYPD Historical Crime Data](https://data.cityofnewyork.us/Public-Safety/NYPD-Complaint-Data-Historic/qgea-i56i): The official data set for the NYPD, all data is provided by the NYPD and is exported as a CSV.

## Files:
Each of these files is located in the Data Folder and then within the following subfolders:

### Raw Data:
#### NCVS_Regional:
    - Number of violent victimizations by crime type, 1996-2022--Sex_ female--Region_ Midwest.csv
    - Number of violent victimizations by crime type, 1996-2022--Sex_ female--Region_ Northeast.csv
    - Number of violent victimizations by crime type, 1996-2022--Sex_ female--Region_ South.csv
    - Number of violent victimizations, 1996-2022--Sex_ female--Region_ West.csv
    - Number of violent victimizations by crime type, 1996-2022--Sex_ female--Region_ South.csv 
    - Number of violent victimizations by crime type, 1996-2022--Sex_ female--Region_ West.csv
    - Number of violent victimizations, 1996-2022--Sex_ female--Region_ Midwest.csv
    - Number of violent victimizations, 1996-2022--Sex_ female--Region_ Northeast (1).csv
    - Number of violent victimizations, 1996-2022--Sex_ female--Region_ South.csv>
#### NCVS_CrimeType:
    - Number of violent victimizations by crime type, 1993-2022--Sex_ female--Age_ 65 or older.csv
    - Number of violent victimizations by crime type, 1993-2022--Sex_ female--Age_ 12 to 14.csv
    - Number of violent victimizations by crime type, 1993-2022--Sex_ female--Age_ 15 to 17.csv
    - Number of violent victimizations by crime type, 1993-2022--Sex_ female--Age_ 18 to 20.csv
    - Number of violent victimizations by crime type, 1993-2022--Sex_ female--Age_ 21 to 24.csv
    - Number of violent victimizations by crime type, 1993-2022--Sex_ female--Age_ 25 to 34.csv
    - Number of violent victimizations by crime type, 1993-2022--Sex_ female--Age_ 35 to 49.csv
    - Number of violent victimizations by crime type, 1993-2022--Sex_ female--Age_ 50 to 64.csv
#### NCVS_Age_Seg:
    - Number of violent victimizations by crime type, 1993-2022--Sex_ female--Age_ 65 or older.csv
    - Number of violent victimizations by crime type, 1993-2022--Sex_ female--Age_ 12 to 14.csv
    - Number of violent victimizations by crime type, 1993-2022--Sex_ female--Age_ 15 to 17.csv
    - Number of violent victimizations by crime type, 1993-2022--Sex_ female--Age_ 18 to 20.csv
    - Number of violent victimizations by crime type, 1993-2022--Sex_ female--Age_ 21 to 24.csv 
    - Number of violent victimizations by crime type, 1993-2022--Sex_ female--Age_ 25 to 34.csv
    - Number of violent victimizations by crime type, 1993-2022--Sex_ female--Age_ 35 to 49.csv
    - Number of violent victimizations by crime type, 1993-2022--Sex_ female--Age_ 50 to 64.csv
#### NYPD:
    - NYPD_Complaint_Data_Current__Year_To_Date__20231024.csv

### Data:
    - NatFlow_AgeTypev1.csv 
    - NCVS_AgeSeg.csv 
    - NCVS_RegionSegv1.csv 
    - NCVS_Regionv1.csv 
    - NYPD_AgeVCrime.csv 
    - NYPDv3.csv

### ML_PreProcess:
    - ML_PreProcess/NCVS_AgeSegML.csv 
    - ML_PreProcess/NCVS_AgeTypeML.csv 
    - ML_PreProcess/NCVS_RegionSegML.csv 
    - ML_PreProcess/NYPD_AgeSegML.csv 
    - ML_PreProcess/NYPDv4ML.csv

### Graphics:
Contains Copites of all the graphs created, include the Geogrpahica Scatter plots for KMeans and DBSCAN.

### HTML:
Each of the Notebooks has been also provide as an HTML file for reference. 

### Paper: 
    - Coffin_ClusteringFemaleVictimology.pdf

## Deployment:

* Crime Victim Data Exploration Analysis.ipynb: A jupyter Notebook developed to exicute data analysis and Exploritory Data Analysis (EDA) as well as note any possible alterations to be made to the data sets for machine learning.

* KMeans Clustering Models Developed for Predictive Modeling.ipynb: A Notebook exploring the use of KMeans on each of the data sets. 

* NYPD DBSCAN Clustering.ipynb: A Jupyter Notebook developing and analizing the results of DBSCAN. 

## Challenges:
During the process of currating the data and building the models, some of the challenges included determining modesl that would work with bother the robust NYPD data set and the anual summaries provided by the NCVS. The NCVS data is a summary of an entire year's worth of responses from victims, with only four possible catagories for personal crime and an even narrower view of age groups. As the NCVS is a survey not all of the data collected has every response, translating to gaps within the data. While this information is cleaned and assembled by the Bureau of Justice Statistics, it creates a challenge when building models. Additionally the data from the NCVS only ranges from 1993 to 2022 with a total of 1334 entries, as compared to the NYPD Data with 2.7 million entries. To address there are two versions of the NYPD data, the first the original filtered data, and the second a condenced data table serving as an anual summary for each age group and crime type with the total number of victims listed - similar to the NCVS data.

While creating models, there were several challenges working the NCVS data into DBSCAN and resulted in only the NYPD data being analysized. Another project utilizing a larger data set from the National Inciden Based Reporting System (NIBRS) data will be required for a more accurate approach to both DBSCAN models and developing a National Baseline as a clearer point of comparison.

## Results:
For detailed findings please refer to [Coffin_ClusteringFemaleVictimology.pdf](\Coffin_ClusteringFemaleVictimology.pdf).  

## Credits:
I would like to acknowlege Stackoverflow, ChatGPT and Andy McDonald's [andymcdgeo](https://github.com/andymcdgeo) tutorials on ["Creating Geospatial Heatmaps With Plotly Express MapBox and Folium in Python - Data Visualisation"](https://www.youtube.com/watch?v=vSGWmZre31A) and ["K-Means Clustering Algorithm with Python Tutorial"](https://www.youtube.com/watch?v=iNlZ3IU5Ffw)

## References:
1. Coffin, A.: Crimevictimanalysis capstone, https://github.com/accoffin12/
CrimeVictimAnalysis_Capstone/tree/main
2. scikit-learn developers: sklearn.cluster.dbscan, https://scikit-learn.
org/stable/modules/generated/sklearn.cluster.DBSCAN.html#
sklearn-cluster-dbscan
3. Doerner, W.G., Lab, S.P.: Measuring Criminal Victimization. Routledge (2017)
4. Doerner, W.G., Lab, S.P.: The Scope of Victimology. Routledge (2017)
5. FBI: Crime data explorer, https://cde.ucr.cjis.gov/LATEST/webapp/#/pages/
explorer/crime/crime-trend
6. FBI: National incident-based reporting system (nibrs), https://www.fbi.gov/
how-we-can-help-you/more-fbi-services-and-information/ucr/nibrs
7. Geron, A.: Unsupervised Learning Techniques. O’Reilly (2017)
8. Geron, A.: Unsupervised Learning Techniques. O’Reilly (2023)
9. of Justice Statistics, B.: Custom graphics: Multi-year trends: Crime type, https:
//ncvs.bjs.ojp.gov/multi-year-trends/crimeType
24 A. Coffin.
10. of Justice Statistics, B.: Data collection: National crime victimization
survey (ncvs), https://www.bjs.gov/index.cfm/content/pub/ascii/content/
data/index.cfm?ty=dcdetail&iid=245
11. Kang, H.W., Kang, H.B.: Prediction of crime occurrence from multi-modal data using deep learning. PLOS ONE 12(4), e0176244 (apr 2017).
https://doi.org/10.1371/journal.pone.0176244, https://doi.org/10.1371%
2Fjournal.pone.0176244
12. NYPD: Nypd complaint data historic, https://data.cityofnewyork.us/
Public-Safety/NYPD-Complaint-Data-Historic/qgea-i56i
