

finalized research question becomes:

Can transfer learning effectiveness in building energy prediction be quantitatively explained by the climatic dissimilarity between source and target regions?

Your hypothesis becomes:

Transfer learning effectiveness decreases as climatic dissimilarity between source and target domains increases.

Now forget the paper for a moment. Think like an engineer building a system for the first time.

The Complete Roadmap

Your project now consists of 7 actual implementation phases.

Phase 0 : Environment Setup
Phase 1 : Data Understanding
Phase 2 : Data Integration
Phase 3 : Feature Engineering
Phase 4 : Baseline Model
Phase 5 : Transfer Learning
Phase 6 : Climate Transferability Experiments
Phase 7 : Analysis and Visualization
PHASE 0 — Environment Setup

You have already:

✓ downloaded:

train.csv
weather_train.csv
building_metadata.csv

✓ opened Google Colab

Now install everything.

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_squared_error

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping

import matplotlib.pyplot as plt
PHASE 1 — Understand The Dataset

This phase exists solely to answer:

What exactly are we feeding into the model?

Load files:

train = pd.read_csv('train.csv')
weather = pd.read_csv('weather_train.csv')
building = pd.read_csv('building_metadata.csv')

Inspect:

train.head()
weather.head()
building.head()

Check:

train.info()
weather.info()
building.info()

Check missing values:

train.isnull().sum()
weather.isnull().sum()
building.isnull().sum()
What you should discover
train.csv

Contains:

building_id
meter
timestamp
meter_reading
weather_train.csv

Contains:

site_id
timestamp
air_temperature
dew_temperature
cloud_coverage
wind_speed
etc
building_metadata.csv

Contains:

building_id
site_id
primary_use
square_feet
year_built
floor_count
PHASE 2 — Select Sites

We decided:

Site	Location	Climate
2	Tempe	2B
4	Berkeley	3C
15	Cornell	6A

Filter electricity only:

train = train[train['meter']==0]

Select desired sites:

sites = [2,4,15]

building = building[
    building.site_id.isin(sites)
]

Extract relevant buildings:

valid_buildings = building.building_id.unique()

train = train[
    train.building_id.isin(valid_buildings)
]

Weather:

weather = weather[
    weather.site_id.isin(sites)
]
PHASE 3 — Merge Dataset

This is your first major contribution in the paper.

First merge:

df = train.merge(
    building,
    on='building_id'
)

Second merge:

df = df.merge(
    weather,
    on=['site_id','timestamp']
)

Now you have:

building_id
site_id
timestamp
meter_reading
square_feet
primary_use
year_built
floor_count
air_temperature
dew_temperature
wind_speed
etc
PHASE 4 — Feature Engineering

This is where you teach the model about time.

Convert:

df['timestamp']=pd.to_datetime(df['timestamp'])

Extract:

df['hour']=df.timestamp.dt.hour
df['month']=df.timestamp.dt.month
df['weekday']=df.timestamp.dt.weekday
df['dayofyear']=df.timestamp.dt.dayofyear

Log transform energy:

df['meter_reading'] = np.log1p(
    df['meter_reading']
)

This dramatically improves performance.

Handle missing values:

df = df.fillna(
    df.median(numeric_only=True)
)

Encode building types:

df = pd.get_dummies(
    df,
    columns=['primary_use']
)
PHASE 5 — Create Climate Datasets

Split:

tempe = df[df.site_id==2]
berkeley = df[df.site_id==4]
cornell = df[df.site_id==15]

Define features:

X = df.drop(
    ['meter_reading',
     'timestamp'],
    axis=1
)

y = df['meter_reading']

Normalize:

scaler = StandardScaler()

X = scaler.fit_transform(X)
PHASE 6 — Build The ANN

Your model:

model = Sequential()

model.add(Dense(
    256,
    activation='relu'
))

model.add(Dense(
    128,
    activation='relu'
))

model.add(Dense(
    64,
    activation='relu'
))

model.add(Dense(
    32,
    activation='relu'
))

model.add(Dense(
    1
))

Compile:

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=1e-5
    ),
    loss='mse'
)
PHASE 7 — Train Source Model

Example:

Tempe → Source

Train:

history = model.fit(
    X_tempe,
    y_tempe,
    epochs=100,
    validation_split=0.2,
    callbacks=[
        EarlyStopping(
            patience=10
        )
    ]
)

Save:

model.save(
    'tempe_model.keras'
)
PHASE 8 — Baseline Cornell

Train from scratch:

Cornell only

Measure:

R²
MAPE
RMSE
Training time
PHASE 9 — Transfer Learning

Load source:

transfer = tf.keras.models.load_model(
    'tempe_model.keras'
)

Freeze:

for layer in transfer.layers[:3]:
    layer.trainable=False

Compile again:

transfer.compile(
    optimizer='adam',
    loss='mse'
)

Train using Cornell:

transfer.fit(
    X_cornell,
    y_cornell
)
PHASE 10 — Data Efficiency Experiment

Train using:

5%
10%
20%
30%
50%
100%

of Cornell.

Store:

R²
MAPE
epochs
time
PHASE 11 — Climate Transferability

Perform:

Tempe → Cornell
Berkeley → Cornell
Scratch → Cornell

Create:

Source	Target	Climate Distance	R²
2B	3C	1	0.92
2B	6A	4	0.88
3C	6A	3	0.90

Then test:

Does transfer performance decrease as climate distance increases?

This is your actual contribution.

PHASE 12 — Explainability

Use:

import shap

Compute:

feature importance
before transfer
after transfer

Question:

Which learned features transfer across climates?

Figures For Paper

Generate:

Figure 1

Dataset pipeline

train
     \
      ---> merge ---> features
     /
weather
     \
building
Figure 2

ANN architecture

Input
 ↓
256
 ↓
128
 ↓
64
 ↓
32
 ↓
Output
Figure 3

Transfer pipeline

Tempe
   ↓
Pretraining
   ↓
Weight Transfer
   ↓
Freeze Layers
   ↓
Cornell
Figure 4

Climate distance vs accuracy

Figure 5

Target data percentage vs performance

What you should do immediately after reading this

Tomorrow's work:

Step 1

Upload:

train.csv
weather_train.csv
building_metadata.csv
Step 2

Run:

head()
info()
shape
isnull()

on all files.

Step 3

Filter:

meter==0
site in [2,4,15]
Step 4

Merge everything.

Step 5

Show:

df.shape
df.head()
df.columns
df.describe()
Step 6

Stop.

Do not build the ANN yet.

Because after the merge we need to perform the exploratory climate analysis, which will determine whether Tempe, Berkeley and Cornell actually produce the climate-transfer hypothesis we designed.

Final "book-style" workflow description for your paper

The proposed research framework investigates the relationship between climatic dissimilarity and transfer learning effectiveness in building energy prediction. The study utilizes the ASHRAE Great Energy Predictor III dataset, which contains building metadata, weather observations, and hourly energy consumption records from multiple climatic regions. Initially, data integration is performed by merging the building metadata, weather, and meter reading datasets using building identifiers, site identifiers, and timestamps. Subsequently, feature engineering techniques are applied to construct a comprehensive feature space, including temporal descriptors such as hour of day, day of week, month, and seasonal indicators, along with building characteristics and meteorological variables. To improve model stability and learning efficiency, energy consumption values are transformed using logarithmic scaling and numerical features are standardized.

The experimental framework is designed around the concept of climate-aware transfer learning. Three representative climate domains are selected, namely Tempe, Arizona (hot-arid climate zone 2B), Berkeley, California (marine climate zone 3C), and Cornell, New York (cold-humid climate zone 6A). Artificial Neural Networks with a five-layer architecture are trained using source climate data to learn generalized building energy consumption patterns. Baseline models are independently trained on the target climate data to establish benchmark performance. Transfer learning experiments are subsequently conducted by transferring learned weights from source climate models to target climate models while selectively freezing early network layers that encode generalized building physics relationships.

To quantify climate-aware transferability, multiple transfer scenarios are evaluated across varying climate distances. The transferred models are further assessed under varying target-data availability conditions, ranging from 5% to 100% of the target dataset, to determine the minimum amount of target data required for successful adaptation. Model performance is evaluated using coefficient of determination (R²), mean absolute percentage error (MAPE), root mean squared error (RMSE), convergence time, and training efficiency. Finally, explainability techniques are employed to analyze the evolution of feature importance before and after transfer learning, enabling interpretation of which physical and environmental characteristics are successfully transferred across climatic domains. Through this framework, the study aims to establish a quantitative relationship between climatic dissimilarity and transfer learning effectiveness in building energy prediction systems.