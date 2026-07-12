# Transfer Learning for Cross-Climate Building Energy Prediction

**Investigating Data Efficiency and Generalization Across Climate Zones**

## Overview
This repository contains the implementation for the research paper exploring cross-climate building energy prediction. The core objective is to investigate whether transfer learning can significantly reduce the amount of target-climate training data required when predicting building energy consumption across different climate zones. 

While existing Net-Zero Energy Building (NZEB) models are often "climate-locked" (performing poorly outside their original training climate), this project leverages **Climate-Aware Transfer Learning** to adapt models across vastly different climates (e.g., from the hot-arid climate of Tempe, AZ to the cold-humid environment of Ithaca, NY) using a fraction of the traditional data requirements.

## Key Contributions
1. **Primary Contribution - Data Efficiency:** Demonstrating that transfer learning can achieve comparable or superior prediction accuracy while using substantially less (e.g., 50% less) target-climate data compared to models trained from scratch.
2. **Secondary Contribution - Climate Transferability:** Quantitatively investigating the impact of climatic dissimilarity (climate distance) on transfer learning effectiveness.
3. **Explainability:** Utilizing SHAP (SHapley Additive exPlanations) to identify which physical and environmental features successfully transfer across climatic domains.

## Dataset
We utilize the **ASHRAE Great Energy Predictor III** benchmark dataset in conjunction with the **Building Data Genome Project 2 (BDGP2)** for site de-anonymization. 
Instead of relying on idealized simulated data (like TRNSYS or EnergyPlus), this project leverages real-world operational data to prove scalability and non-linear adaptability.

### Selected Domains
| Site ID | Location (De-anonymized) | Climate Zone | Climate Type |
| :---: | :--- | :---: | :--- |
| **2** | Arizona State University (Tempe, AZ) | **2B** | Hot-Arid |
| **4** | UC Berkeley (Berkeley, CA) | **3C** | Marine |
| **15** | Cornell University (Ithaca, NY) | **6A** | Cold-Humid |

## Evaluation Metrics
Because energy prediction is a *regression* task (predicting a continuous kWh value), standard classification accuracy does not mathematically apply. The models are rigorously evaluated using standard regression metrics:
- **$R^2$ Score**: Proportion of variance explained (Primary metric).
- **RMSE (Root Mean Squared Error)**: Measures the average magnitude of absolute error.
- **MAPE (Mean Absolute Percentage Error)**: Measures the relative percentage error.
- **Accuracy Proxy**: To satisfy general reporting requirements or reviewers asking for "accuracy", an intuitive proxy metric is calculated as `100 - (MAPE * 100)`.

## Methodology & Pipeline
The workflow is structured into discrete experimental phases:
- **Phase 1-3 (Data Integration):** Load and merge `train.csv`, `weather_train.csv`, and `building_metadata.csv` (filtered to electricity meters for the chosen sites).
- **Phase 4 (Feature Engineering):** Extract temporal cyclic features, apply logarithmic scaling to energy targets (`np.log1p`), and standardize numerical variables.
- **Phase 6 (Architecture):** A 5-layer Artificial Neural Network (ANN) `[256 -> 128 -> 64 -> 32 -> 1]` acts as the surrogate model.
- **Phase 7-9 (Transfer Learning):** Pre-train the ANN on a source climate (e.g., Tempe). Transfer the weights to a target climate model, freeze the early layers (which encode generalized building physics), and fine-tune using target data.
- **Phase 10 (Data Efficiency Experiment):** Train models using escalating subsets (5%, 10%, 20%, 30%, 50%, 100%) of the target dataset to plot performance vs. data availability.
- **Phase 11 (Climate Analysis):** Construct a transferability matrix mapping climate distance against $R^2$ performance gains.
- **Phase 12 (SHAP Explainability):** Extract global feature importance pre- and post-transfer.

## Project Structure
- `notebooks/NZEB_Climate_Transfer.ipynb` - The primary experimentation notebook (Kaggle/Colab compatible).
- `src/` - Modularized Python codebase containing data loaders, feature engineering pipelines, model architectures, and training scripts.
- `Reference Docs/` - Project documentation, planning frameworks, and theoretical justifications.

## Getting Started (Kaggle)
This codebase is fully configured for Kaggle environments to bypass local hardware restrictions. 
1. Compress the `src/` folder and `merged_ashrae.csv` into a `.zip` archive.
2. In your Kaggle Notebook, click **Add Data** -> **Upload Data** -> **New Dataset**.
3. Upload the `.zip` archive. Kaggle will automatically mount it to `/kaggle/input/`.
4. Run the `NZEB_Climate_Transfer.ipynb` notebook. The script will automatically detect the Kaggle environment, locate your uploaded dataset, and run the pipeline without requiring Google Drive mounts.