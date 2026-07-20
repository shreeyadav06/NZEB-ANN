# PPT Content (Review 2 Presentation)

## Slide 1: Title Slide
*   **Title:** Climate-Aware Transfer Learning for Net Zero Energy Buildings (NZEB)
*   **Sub-title:** Summer Internship (CSXXXX) - Review 2
*   *(Fill in your Team Members and Guide Details)*

## Slide 2: Title and Problem Definition
*   **Problem:** Newly built NZEBs suffer from a "cold start" data scarcity problem. They lack historical data, making it impossible for standard machine learning models to accurately predict and optimize their energy consumption.
*   **Limitations of existing systems:** Current predictive models require years of building-specific data to function accurately. They fail when applied to brand new buildings. Traditional physics-based simulations (like EnergyPlus) are highly accurate but computationally expensive and slow.
*   **Novelty:** Proposing a Transfer Learning (TL) approach to transfer energy consumption patterns from data-rich climates (Source) to data-scarce new buildings in different climates (Target), completely bypassing the data collection waiting period.

## Slide 3: Aim and Objectives
*   **Aim:** To develop and validate a Transfer Learning framework that effectively predicts building energy consumption in data-scarce environments by leveraging Artificial Neural Networks (ANNs) pre-trained on diverse ASHRAE climate zones.
*   **Objectives:**
    *   To process and encode temporal energy data across three distinct climate zones (Tempe, Berkeley, Cornell).
    *   To design a baseline deep feed-forward Artificial Neural Network (ANN) from scratch.
    *   To implement a Transfer Learning pipeline by freezing the initial layers of a pre-trained model and fine-tuning the final layer on target data.
    *   To identify the exact "crossover point" (data fraction percentage) where Transfer Learning loses its advantage over training from scratch.

## Slide 4: Introduction & Scope
*   **Basic Introduction:** As the world moves towards Net Zero Energy Buildings to combat climate change, accurate energy prediction is critical for grid stability and building optimization.
*   **Scope and Context:** The project focuses on utilizing the ASHRAE Great Energy Predictor III dataset to predict electricity consumption (Meter 0) across highly diverse climate zones (Zone 2B, 3C, and 6A).
*   **Key Challenges:** Handling massive datasets (800k+ rows per site), managing skewed data, and ensuring models do not overfit when simulating extreme data scarcity (e.g., training on only 5% of data).

## Slide 5: Literature Survey
*   **Summary:** Existing literature heavily focuses on using AI to replace slow, physics-based building simulations. Optimization papers use AI surrogate models to evaluate thousands of designs instantly (e.g., using Genetic Algorithms). Prediction papers use LSTMs for forecasting based on historical data.
*   **Research Gap:** While AI is proven to reduce computational cost compared to traditional simulations, existing literature assumes ample historical data is available. There is a critical gap in predictive modeling for *new* buildings immediately post-construction.
*   **Justification:** Our proposed work bridges this gap by introducing Transfer Learning to eliminate the dependency on long-term historical data generation.

## Slide 6: Requirement Specification
*   **Software Requirements:**
    *   **Languages & Libraries:** Python 3.x, TensorFlow, Keras, Pandas, Scikit-Learn, Matplotlib/Seaborn (for visualization).
    *   **Environment:** Kaggle Notebooks (for headless cloud execution).
*   **Hardware Requirements:**
    *   **Compute:** Kaggle Cloud GPUs (Dual NVIDIA T4 GPUs) for parallel processing and overcoming local memory constraints.

## Slide 7: Proposed Methodology (System Architecture)
```mermaid
graph TD
    subgraph Pipeline [Data Pipeline]
        A[Parse ASHRAE Dataset] --> B[Log Transformations]
        B --> C[Sine/Cosine Cyclical Encoding]
        C --> D[Standard Scaling]
    end
    
    D -->|Parallel Track 1| E
    D -->|Parallel Track 2| G

    subgraph Phase1 [Phase 1: Baseline]
        E[Train 4-layer ANN from Scratch<br/>Target Climate] --> F[5-Fold Cross Validation]
    end
    
    subgraph Phase2 [Phase 2: Transfer Learning]
        G[Train ANN on Source Climate] --> H[Freeze Layers 1-3]
        H --> I[Transfer to Target Climate]
        I --> J[Fine-tune Last Layer]
    end
    
    F -.-> K
    J -.-> K
    
    subgraph Phase3 [Phase 3: Data Efficiency Loop]
        K{Compare Performance} --> L[Vary Target Data Fractions<br/>5% to 100%]
    end
```
*   **Data Pipeline:** Parsing ASHRAE dataset -> Log Transformations for skewed data -> Sine/Cosine cyclical encoding for temporal features -> Standard Scaling.
*   **Phase 1 (Baseline):** Train a 4-hidden-layer ANN `[256, 128, 64, 32]` from scratch on the target climate (5-Fold Cross Validation).
*   **Phase 2 (Transfer):** Train ANN on a source climate (e.g., Tempe). Freeze layers 1 to 3. Transfer to the target climate (e.g., Cornell) and fine-tune only the last layer.
*   **Phase 3 (Data Efficiency Loop):** Restrict target training data to fractions (5%, 10%, 20%, 30%, 50%, 100%) and compare Scratch vs. Transfer performance.

## Slide 8: Module-wise Implementation (Since Review 1)
*   **Module 1: Cloud Orchestration:** Successfully migrated the entire training pipeline to Kaggle Cloud GPUs, optimizing batch sizes (256) to handle 800k+ rows per site without out-of-memory errors.
*   **Module 2: The Data Efficiency Loop:** Implemented the core experiment suggested by the mentor—iteratively training models on restricted data fractions to simulate the lifespan of a new building.
*   **Module 3: Statistical Validation:** Ran all Transfer and Scratch pipelines across 3 different random seeds (42, 123, 456) to ensure the findings were statistically significant and not flukes.

## Slide 9: Experimental Results / Output
*   *Place your "Primary Data Efficiency" graph here.*
*   **Key Result:** At extreme scarcity (5% data), Transfer Learning achieved an R² of 0.9936 compared to Scratch's 0.8124 (a massive +0.1812 boost).
*   *Place your "Convergence Plot" here.*
*   **Key Result:** The crossover point was successfully identified precisely at 30% data availability, where the Scratch model finally catches up to the frozen Transfer model.

## Slide 10: Conclusion
*   The project successfully proves that Transfer Learning is a highly effective, computationally efficient solution for the NZEB "cold start" problem. 
*   By utilizing pre-trained weights from data-rich climates, new buildings can achieve highly accurate energy predictions (0.99 R²) with only a tiny fraction (5%) of required historical data.

## Slide 11: Future Work
*   Extend the Transfer Learning architecture to utilize Recurrent Neural Networks (LSTMs or GRUs) to capture longer-term sequential dependencies.
*   Test transferability across different building types (e.g., transferring from a commercial office building to an educational facility) rather than just across climates.
