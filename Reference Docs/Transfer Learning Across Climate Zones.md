

NZEB Design using ML Techniques  
\#\# FINAL LIT. REVIEW AND POTENTIAL TOPICS

\#\# Brief  
The collection of literature papers is not really about buildings; it is mainly about using AI  
and machine learning to replace or accelerate slow, traditional, physics-based building  
energy simulations. Existing literature typically involves generating large amounts of  
simulation data, training ML models such as ANN, LSTM, Random Forest, or SVM on that  
data, and then using optimization algorithms to quickly find the best building designs.  
The common workflow across most papers is:  
simulation → dataset generation → ML model training → optimization → best design  
selection.  
Final Summary of 14 Papers  
The key finding is the distinction between prediction and optimization studies.  
Prediction-focused papers use models like LSTM to forecast future energy consumption,  
temperature, or comfort levels over time. Optimization-focused papers use ANN-based  
surrogate models that imitate expensive simulations and work together with algorithms  
like Genetic Algorithms, Grey Wolf Optimization, or NSGA-III to search for the best building  
configurations.  
A major finding across the literature is that AI significantly reduces computational cost.  
Instead of running thousands of expensive simulations, researchers train a model once and  
then use it to make near-instant predictions. This allows optimization algorithms to  
evaluate many design alternatives much faster while maintaining reasonable accuracy.  
Most studies agree that AI is useful because building systems are highly nonlinear and  
difficult to model using traditional methods. However, researchers differ in the specific

techniques they prefer, with some focusing on ANNs, others on LSTMs, Random Forests,  
SVMs, or different optimization algorithms. Despite these differences, the overall  
conclusion remains the same: AI-assisted optimization is effective for improving building  
performance.  
The most important research gaps identified are the lack of cross-climate generalization,  
heavy reliance on simulation-generated data, and unrealistic assumptions about human  
behavior. Most models are trained and tested in the same region, rarely validated on real  
buildings, and often assume predictable occupant schedules. These limitations raise  
concerns about how well the models perform in real-world situations.

\#\# Found Research Gaps:  
From an AI/ML perspective, the strongest future research directions include transfer  
learning across climate zones, explainable AI for energy prediction, evaluating  
synthetic-versus-real data performance, benchmarking modern architectures such as  
transformers against traditional models, and studying model robustness under future  
climate scenarios. Among these, transfer learning across climate regions appears to be the  
most promising balance of novelty, practicality, and publication potential.  
AI in these papers is primarily used as a surrogate model that approximates expensive  
building simulations. Most datasets are simulation-generated rather than collected from  
real buildings. Generalization across climates and real-world deployment remain major  
unresolved challenges. Current literature relies heavily on traditional ML techniques and  
has limited exploration of modern methods such as transformers, transfer learning,  
reinforcement learning, federated learning, or digital twins.  
The strongest publication opportunity is developing models that adapt effectively across  
different climate zones using minimal new data.

\#\# 2

Final Research Paper Options, Outlines and Testable  
\#\# Hypotheticals  
\#\# Transfer Learning Across Climate Zones  
Most ML models are "climate-locked." A model trained for a temperate climate (like  
France) often fails or requires massive retraining to work in a polar climate (like  
\#\# Canada)  
. Almost every paper trains and tests in the same climate region.  
"Can a model trained in one climate be adapted to another using only a small amount of new  
data?"  
\#\# Hypothesis:  
“An ANN energy-prediction model pre-trained on a large "Hot-Arid" dataset can reach 90%  
accuracy in a "Tropical" climate using 50% less new training data compared to a model  
trained from scratch.“  OR  
“A transfer-learning-based surrogate model can achieve similar prediction accuracy to a  
fully retrained model while using significantly less training data from a new climate zone.”  
Proposed Methodology: Use an open-source dataset (like the ASHRAE Great Energy  
Predictor) to collect building-energy datasets from multiple climate regions. Train a simple  
Random Forest/ANN model on a cold city data. Test its accuracy on a warm city. Perform a  
"fine-tuning" step where you add 10% of the warm city's data to the training set and  
observe the accuracy jump.  
Compare: Baseline ANN, Transfer Learning ANN, Fine-tuned ANN  
Measure: MAE, RMSE, R²

