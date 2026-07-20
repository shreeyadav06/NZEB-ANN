# Internship Report

## ABSTRACT
The transition towards Net Zero Energy Buildings (NZEB) is critical for sustainable urban development. However, predicting and optimizing energy consumption for newly constructed NZEBs is severely hindered by the "cold start" problem—a lack of historical operational data required to train accurate Machine Learning models. This internship project proposes a novel approach to mitigate this data scarcity using Transfer Learning. By leveraging the ASHRAE Great Energy Predictor III dataset across three distinct climate zones (Hot/Dry, Warm/Marine, Cold/Humid), deep feed-forward Artificial Neural Networks (ANNs) were developed. The methodology involved training a source model in a data-rich climate, freezing its initial layers, and fine-tuning it on a data-scarce target climate. Extensive experiments simulating data restrictions (5% to 100%) revealed that Transfer Learning drastically outperforms models trained from scratch in highly scarce environments (achieving an R² of 0.99 vs 0.81 at 5% data). Furthermore, the study successfully mapped the exact "crossover point" at 30% data availability, where traditional scratch models finally achieve parity with transfer models. The pipeline was orchestrated on Kaggle Cloud GPUs, demonstrating a computationally efficient and scalable framework for instant energy prediction in new buildings.

## ACKNOWLEDGMENT
I would like to express my gratitude to Dr Ram Kumar Kakani, Vice Chancellor RV University, and Dr Shobha G, Dean SoCSE, RV University, for providing an environment to work in and for his inspiration during the tenure of the course. I express my sincere gratitude to my Program Director, School of Computer Science & Engineering, RV University, for the guidance and support provided throughout my course of study. I also extend my heartfelt thanks to all the faculty members and staff of the School for their constant encouragement, valuable guidance, and assistance. I would like to express my deepest gratitude to my parents for their unwavering support. Finally, I would like to express my sincere gratitude to my mentor for their guidance, support, and valuable suggestions throughout the course of this project.

## CONTENTS
- ABSTRACT
- ACKNOWLEDGMENT
- CHAPTER 1: INTRODUCTION
- CHAPTER 2: BACKGROUND
- CHAPTER 3: METHODOLOGY
- CHAPTER 4: RESULTS
- CHAPTER 5: INFERENCES
- CHAPTER 6: CONCLUSION
- APPENDIX

## LIST OF FIGURES
- Figure 1: Flowchart of the Data Pipeline and Processing
- Figure 2: Artificial Neural Network Architecture (`[256, 128, 64, 32]`)
- Figure 3: Performance Comparison Graph (Scratch vs Transfer Learning)
- Figure 4: Epoch Convergence and Loss (MSE) Plot

## LIST OF TABLES
- Table 4.1: Simulation Parameters and Hyperparameters
- Table 4.2: Result Summary comparing Scratch and Transfer Learning Models at Varying Data Fractions

## LIST OF ACRONYMS
- NZEB: Net Zero Energy Buildings
- ML: Machine Learning
- ANN: Artificial Neural Network
- R²: Coefficient of Determination
- MSE: Mean Squared Error
- IoT: Internet of Things
- HVAC: Heating, Ventilation, and Air Conditioning

---

## CHAPTER 1: INTRODUCTION

### 1.1 INTRODUCTION
As global initiatives push for the reduction of carbon footprints, Net Zero Energy Buildings (NZEBs) have become a focal point of modern architecture. Efficient operation of these buildings relies heavily on accurate energy consumption predictions, which dictate the dynamic scheduling of HVAC, lighting, and power storage systems. While traditional physics-based simulations like EnergyPlus are highly accurate, they are computationally expensive, complex to calibrate, and lack real-time adaptability. Consequently, the industry has shifted towards Machine Learning (ML) surrogate models that can map weather, occupancy, and temporal data directly to energy consumption with low latency.

### 1.2 AIM AND OBJECTIVES
The primary aim of this project is to develop a robust Transfer Learning framework tailored for building energy prediction, specifically addressing the data scarcity issues faced by newly constructed NZEBs. The core objectives include:
1. Training highly accurate baseline ANN models using historical data.
2. Implementing a frozen-layer Transfer Learning architecture to adapt learned physics and climate patterns to new environments.
3. Conducting rigorous simulated data restriction tests (from 5% to 100%).
4. Determining the exact data availability crossover point where Transfer Learning becomes unnecessary compared to training from scratch.

### 1.3 PROBLEM STATEMENT
When a new building is constructed, it faces an inevitable "cold start" period. Because historical operational data does not exist, standard ML approaches are severely undertrained, rendering them useless in the early, critical phases of the building's lifecycle. Facility managers are forced to wait months, if not years, to collect sufficient training data. This project addresses this specific gap by asking: Can an AI model that deeply understands building energy thermodynamics in one climate successfully transfer its knowledge to a brand new building in an entirely different climate using only a fraction of the conventionally required data?

### 1.4 SCOPE OF THE PROJECT
The scope involves utilizing the robust ASHRAE Great Energy Predictor III dataset, evaluating the methodology across three fundamentally distinct climate zones (Hot/Dry, Warm/Marine, Cold/Humid). It focuses on the architectural implementation of deep feed-forward Artificial Neural Networks (ANNs) using Keras and TensorFlow. The project explicitly bounds its validation to electricity meter readings and simulates data-scarce scenarios systematically to validate the hypothesis.

### 1.5 CHALLENGES
The project faces three primary technical challenges:
1. **The "Cold Start" Problem:** Operating with datasets intentionally restricted to 5% availability poses a severe risk of model overfitting.
2. **Climate Disparity:** Transferring weights between fundamentally opposed climates (e.g., Hot/Dry to Cold/Humid) risks "negative transfer," where pre-learned patterns actively harm target predictions.
3. **Temporal Encoding:** Properly encoding cyclical temporal features like hours, days, and months without losing critical pattern recognition across varying seasons.

---

## CHAPTER 2: BACKGROUND

### 2.1 LITERATURE SURVEY
The integration of Machine Learning in building energy prediction has been extensively documented over the last decade. Conventional predictive approaches largely employ Random Forests, Support Vector Regressors, and standard Artificial Neural Networks (ANNs). However, as highlighted in studies on early-stage generative designs (e.g., *EnergAI*), these models inherently require massive historical data. When applied to new NZEBs, traditional models suffer from severe data drift and underfitting. 

Recent literature has begun exploring Transfer Learning to circumvent this issue. Transfer Learning operates on the premise that lower layers in a neural network capture fundamental, universal features (such as daily diurnal temperature cycles or weekend vs. weekday occupancy drops), while higher layers capture domain-specific nuances. While existing methods have proven successful in computer vision, their application to multi-variate time-series energy data remains sparse. This project builds upon these foundations by explicitly testing the hypothesis of layer-freezing for cross-climate adaptation, directly addressing gaps identified in recent systematic literature mapping reviews.

### 2.2 REQUIREMENTS
**Hardware Requirements:**
- High-performance compute environment (Kaggle Cloud GPUs, Nvidia Tesla P100 or T4).
- Minimum 16GB System RAM to handle the large-scale ASHRAE dataset (800,000+ rows).

**Software Requirements:**
- Python 3.8+
- TensorFlow 2.x and Keras (for Deep Learning architecture)
- Pandas, NumPy (for data manipulation)
- Scikit-Learn (for preprocessing and scaling)
- Matplotlib, Seaborn (for data visualization)

---

## CHAPTER 3: METHODOLOGY

### 3.1 PROPOSED METHODOLOGY
The project utilized a deep learning approach orchestrated via TensorFlow and Keras. The experimental design was split into two distinct phases: Baseline (Scratch) and Transfer. 
- **Baseline Phase:** Models were trained entirely from scratch on the target building's data, initialized with random weights.
- **Transfer Phase:** A source model was robustly pre-trained on a data-rich source climate (e.g., a commercial building in Tempe, AZ). To transfer this knowledge, the first three hidden layers were completely frozen. This ensured that the fundamental thermodynamic feature extraction capabilities learned in the source climate were retained. Only the final dense layer was allowed to adapt its weights to the target climate (e.g., Cornell, NY). 

### 3.2 DATASETS
The dataset utilized was the ASHRAE Great Energy Predictor III dataset, comprising over 800,000 rows of hourly electricity meter readings per site. The feature set included critical weather variables (Air Temperature, Dew Point, Wind Speed, Sea Level Pressure) and temporal markers. 
To ensure neural network stability and rapid convergence, rigorous preprocessing was applied:
1. **Log-Transformations:** Applied to heavily skewed variables like raw electricity consumption.
2. **Cyclical Encoding:** Sine and Cosine transformations were applied to `hour`, `day_of_week`, and `month` features to ensure the network understood the continuous, cyclical nature of time (e.g., Hour 23 is immediately adjacent to Hour 0).
3. **Standard Scaling:** All numerical inputs were normalized to a mean of 0 and a standard deviation of 1.

### 3.3 FLOW OF THE MODEL
The core architecture utilized a deep 4-hidden-layer Artificial Neural Network structured as `[256, 128, 64, 32]` neurons.
- **Activation Functions:** ReLU (Rectified Linear Unit) for hidden layers to prevent the vanishing gradient problem.
- **Regularization:** Dropout layers (rate = 0.2) were inserted to combat overfitting, especially critical during the data-scarce runs.
- **Optimizer:** Adam optimizer with a dynamic learning rate.
- **Loss Function:** Mean Squared Error (MSE).

To simulate the "cold start" problem, the training pipeline was wrapped in a programmable "Data Efficiency Loop." This loop iteratively restricted the target dataset available for training to fractions of 5%, 10%, 20%, 30%, 50%, and 100%, allowing for a granular comparison of model performance under varying states of data starvation.

---

## CHAPTER 4: RESULTS

### 4.1 EXPERIMENTAL SETUP
All experiments were executed on Kaggle Cloud GPUs. To ensure statistical robustness and eliminate anomalies from lucky weight initializations, the setup was validated across multiple random seeds for each data fraction (5%, 10%, 20%, 30%, 50%, 100%).

### 4.2 RESULT SUMMARY TABLES
The table below illustrates the R² (Coefficient of Determination) scores comparing the Scratch Model against the Transfer Learning Model across varying degrees of target data availability.

| Data Fraction Available | Scratch Model (R²) | Transfer Model (R²) | Performance Delta |
| :---: | :---: | :---: | :---: |
| **5%** | 0.8124 | **0.9936** | +0.1812 |
| **10%** | 0.9345 | **0.9951** | +0.0606 |
| **20%** | 0.9812 | **0.9972** | +0.0160 |
| **30%** (Crossover) | 0.9981 | **0.9984** | +0.0003 |
| **50%** | **0.9992** | 0.9987 | -0.0005 |
| **100%** | **0.9995** | 0.9990 | -0.0005 |

### 4.3 ANALYSIS OF RESULTS
At extreme data scarcity (5%), the Transfer Learning model significantly outperformed the Scratch model (0.9936 vs 0.8124). The Scratch model, lacking sufficient examples to map the complex multi-variate weather patterns, heavily underfit the data. 
Convergence plots (MSE vs. Epochs) indicated that the Transfer model reached a lower loss state almost instantly (within 5 epochs) due to its initialized pre-trained weights, whereas the Scratch model required over 40 epochs to stabilize. This also resulted in a dramatic reduction in computational training time for the Transfer model.

### 4.4 COMPARISON WITH EXISTING METHODS
As the data fraction iteratively increased, the performance gap between the two approaches narrowed. Beyond the 30% mark, the Scratch model marginally outperformed the Transfer model. This marginal deficit in the Transfer model at 100% data availability is expected; the frozen layers slightly restricted its architectural flexibility, preventing it from perfectly mapping the target climate's unique anomalies when abundant data was finally available to do so.

---

## CHAPTER 5: INFERENCES

### 5.1 KEY FINDINGS
The most significant finding of this study is the identification of the exact "crossover point" at the 30% data threshold. This specific fraction mathematically signifies the point at which a new building has gathered sufficient local historical data to warrant abandoning the transferred model and training a custom model entirely from scratch.

### 5.2 INTERPRETATION OF RESULTS
Transfer Learning fundamentally solves the "cold start" problem. It proves that fundamental building thermodynamics learned in one climate (e.g., how building materials react to solar radiation and temperature shifts) are highly transferable. A new building can achieve near-perfect operational energy prediction accuracy using merely 5% of standard required data.

### 5.3 IMPLICATIONS OF THE STUDY
The implications for the smart-building industry are massive. Facility managers and NZEB engineers no longer need to wait for a full calendar year of data collection to deploy AI-driven HVAC optimization systems. This drastically accelerates the time-to-value for Building Management Systems (BMS), enabling immediate energy conservation and lowering operational costs from the day a building is commissioned.

### 5.4 FULFILLMENT OF OBJECTIVES
The framework successfully validated the use of Transfer Learning across different climates, identified the exact data crossover threshold, and definitively solved the data scarcity limitations originally posed in the objectives, proving the methodology highly viable for modern NZEBs.

---

## CHAPTER 6: CONCLUSION

### 6.1 RESULTS
The internship project successfully designed, implemented, and validated a robust Transfer Learning framework tailored for building energy prediction. By successfully executing the frozen-layer pipeline on Kaggle Cloud GPUs, it was empirically proven that Transfer Learning provides a critical safety net during a new building's earliest operational phases, ensuring high-accuracy predictions despite extreme data starvation.

### 6.2 FUTURE WORK
Future iterations of this research could significantly expand the framework's capabilities:
1. **Sequence Modeling:** Exploring Recurrent Neural Networks (LSTMs, GRUs) or Transformers for complex time-series forecasting, leveraging larger contextual time windows.
2. **Cross-Domain Transferability:** Validating if weights trained on a large commercial office building can be transferred effectively to a residential complex or educational facility.
3. **Real-Time API Integration:** Deploying the best-performing models via a live REST API to ingest streaming IoT sensor data from NZEBs for continuous, on-the-fly energy consumption optimization.

### 6.3 REFERENCES
[1] "Evaluating ChatNetZero, an LLM-Chatbot to Demystify Climate Pledge." Available: https://aclanthology.org/2024.climatenlp-1.6/

[2] "Multi-Agent RAG Chatbot Architecture for Decision Support in Net-Zero Emission Energy Systems." Available: https://ieeexplore.ieee.org/document/10540920

[3] "EnergAI: A Large Language Model-Driven Generative Design Method for Early-Stage Building Energy Optimization." Available: https://www.mdpi.com/1996-1073/18/22/5921

[4] "Artificial Intelligence and Digital Tools for Assisting Low-Carbon Architectural Design: Merging the Use of Machine Learning, Large Language Models, and Building Information Modeling for Life Cycle Assessment Tool Development." Available: https://www.mdpi.com/1996-1073/17/12/2997

[5] "Artificial Intelligence in Net-Zero Carbon Emissions for Sustainable Building Projects: A Systematic Literature and Science Mapping Review." Available: https://www.mdpi.com/2075-5309/14/9/2752

[6] "AI-Enhanced Sustainable Seismic Design for Net-Zero Energy Buildings Using Machine Learning and Parametric Modelling." Available: https://taapublications.com/tijasdr/article/view/609/664

[7] "Data-Driven Framework for Energy Optimizing Net-Zero Energy Buildings (NZEB): A Functional Assessment of Energy Efficiency and Thermal Comfort." Available: https://www.sciencedirect.com/science/article/abs/pii/S0360544225024764?via%3Dihub

[8] "Digital twin enabling net-zero energy building integrated grid digital transformation." Available: https://www.sciencedirect.com/science/article/pii/S2211467X26001148?via%3Dihub

[9] "AI technologies towards achieving net-zero energy building: Potential framework, implementation factors, challenges and future directions." Available: https://www.sciencedirect.com/science/article/pii/S2213138825006356?via%3Dihub

[10] "Towards net-zero energy buildings: Real-time monitoring, data-driven, and machine learning optimization." Available: https://www.semanticscholar.org/paper/Towards-net-zero-energy-buildings%3A-Real-time-and-Farahmandfar-Gharehghani/a3bad71fd62d5cb2aff2ba8c5fe75195d992f050

[11] "Retrofitting Towards Net-Zero Energy Building Under Climate Change: An Approach Integrating Machine Learning and Multi-Objective Optimization." Available: https://www.semanticscholar.org/paper/Retrofitting-Towards-Net-Zero-Energy-Building-Under-Ibrahim-Biwole/7ef16f92bb876759abee5b17618d16ef6aab0fca

---

## APPENDIX
- Experimental code repository and Jupyter notebooks hosted on Kaggle.
