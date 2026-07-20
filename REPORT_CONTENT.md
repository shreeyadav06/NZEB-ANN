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
- (To be added based on report figures)

## LIST OF TABLES
- (To be added based on report tables)

## LIST OF ACRONYMS
- NZEB: Net Zero Energy Buildings
- ML: Machine Learning
- ANN: Artificial Neural Network

---

## CHAPTER 1: INTRODUCTION

### 1.1 INTRODUCTION
As global initiatives push for the reduction of carbon footprints, Net Zero Energy Buildings (NZEBs) have become a focal point of modern architecture. Efficient operation of these buildings relies heavily on accurate energy consumption predictions. While traditional physics-based simulations like EnergyPlus are highly accurate, they are computationally expensive and time-consuming. Consequently, the industry has shifted towards Machine Learning (ML) surrogate models.

### 1.2 AIM AND OBJECTIVES
The primary aim of this project is to develop a Transfer Learning framework tailored for building energy prediction, specifically addressing the data scarcity issues faced by newly constructed NZEBs. The objectives include training baseline models, implementing frozen-layer transfer learning, and determining the exact data availability crossover point where transfer learning becomes unnecessary.

### 1.3 PROBLEM STATEMENT
When a new building is constructed, it faces a "cold start" period where historical operational data does not exist, rendering standard ML approaches useless in the early, critical phases of the building's lifecycle. This project addresses this specific gap by asking: Can an AI model that understands building energy dynamics in one climate transfer its knowledge to a brand new building in an entirely different climate?

### 1.4 SCOPE OF THE PROJECT
The scope involves utilizing the ASHRAE dataset across three climate zones. It focuses on the architectural implementation of deep feed-forward Artificial Neural Networks (ANNs) and validating their performance under simulated data-scarce scenarios (5% to 100% data availability).

### 1.5 CHALLENGES
The main challenge lies in the "cold start" problem, neural network stability under extreme data scarcity, and properly encoding cyclical temporal features like hours, days, and months without losing critical pattern recognition.

---

## CHAPTER 2: BACKGROUND

### 2.1 LITERATURE SURVEY
(Extensive literature review content on Transfer Learning in NZEBs and ML surrogate models goes here)

### 2.2 REQUIREMENTS
The project requires a deep learning environment orchestrated via TensorFlow and Keras, alongside high-performance compute resources such as Kaggle Cloud GPUs.

---

## CHAPTER 3: METHODOLOGY

### 3.1 PROPOSED METHODOLOGY
The project utilized a deep learning approach orchestrated via TensorFlow and Keras. The experimental design was split into Baseline and Transfer phases. Baseline models were trained entirely from scratch on the target data. For the Transfer phase, a model was pre-trained on a source climate (e.g., Tempe). The first three layers were frozen to retain fundamental feature extraction capabilities, and only the final dense layer was allowed to adapt to the target climate (e.g., Cornell).

### 3.2 DATASETS
The dataset comprised over 800,000 rows of electricity meter readings per site from the ASHRAE dataset. To ensure neural network stability, rigorous preprocessing was applied, including log-transformations for skewed variables, Standard Scaling, and Sine/Cosine transformations.

### 3.3 FLOW OF THE MODEL
The core architecture utilized a 4-hidden-layer Artificial Neural Network `[256, 128, 64, 32]` with ReLU activations and an Adam optimizer. To simulate data scarcity, the training pipeline was wrapped in a "Data Efficiency Loop," which restricted the target data to various fractions (5% to 100%).

---

## CHAPTER 4: RESULTS

### 4.1 EXPERIMENTAL SETUP
The setup was validated across multiple random seeds for statistical robustness, testing fractions at 5%, 10%, 20%, 30%, 50%, and 100%.

### 4.2 RESULT SUMMARY TABLES
At extreme data scarcity (5%), the Transfer Learning model achieved an R² score of 0.9936, significantly outperforming the Scratch model, which achieved an R² of 0.8124. At 30%, the Scratch model (R² 0.9981) caught up to the Transfer model (R² 0.9984).

### 4.3 ANALYSIS OF RESULTS
Convergence plots indicated that the Transfer model reached a lower loss state much faster in early epochs due to its initialized pre-trained weights. As the data fraction was iteratively increased, the performance gap between the two approaches narrowed.

### 4.4 COMPARISON WITH EXISTING METHODS
Beyond 30%, the Scratch model marginally outperformed the Transfer model, as the frozen layers in the Transfer model slightly restricted its ability to perfectly map the target climate when 100% of the data was finally available.

---

## CHAPTER 5: INFERENCES

### 5.1 KEY FINDINGS
The critical "crossover point" was identified at the 30% data threshold. This exact fraction signifies when sufficient local data has been gathered to warrant training a model entirely from scratch.

### 5.2 INTERPRETATION OF RESULTS
Transfer Learning allows a new building to achieve near-perfect energy prediction accuracy using only 5% of standard required data.

### 5.3 IMPLICATIONS OF THE STUDY
This drastically accelerates the time-to-value for smart building energy management systems, effectively neutralizing the cold start problem for NZEBs.

### 5.4 FULFILLMENT OF OBJECTIVES
The framework successfully validated the use of Transfer Learning across different climates and definitively solved the data scarcity limitations originally posed in the objectives.

---

## CHAPTER 6: CONCLUSION

### 6.1 RESULTS
The internship project successfully designed, implemented, and validated a Transfer Learning framework tailored for building energy prediction on Kaggle Cloud GPUs.

### 6.2 FUTURE WORK
Future iterations could explore sequence models (LSTMs) for time-series forecasting, cross-domain transferability across building types, and deployment via real-time APIs.

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
