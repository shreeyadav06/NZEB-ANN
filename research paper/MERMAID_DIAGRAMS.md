# System Architecture & Neural Network Topology (Mermaid Diagrams)

This document contains the official Mermaid diagram definitions used for Figure 1 and Figure 2 in the research paper:
*"Transfer Learning for Cross-Climate Building Energy Prediction: Investigating Data Efficiency and Generalization Across Climate Zones"*

---

## 1. System Pipeline Architecture (Figure 1)

```mermaid
flowchart LR
    subgraph S1 ["1. Data Ingestion & Integration"]
        direction TB
        D1["ASHRAE GEPIII & BDGP2"] --> D2["Filter Meter 0: Electricity"]
        D2 --> D3["Multi-Site Split: Sites 2, 4, 15"]
        D3 --> D4["Merge Weather & Metadata"]
    end

    subgraph S2 ["2. Feature Engineering"]
        direction TB
        F1["Cyclic Sine/Cosine Time Transforms"] --> F2["Log Target Transform: ln(1 + E)"]
        F2 --> F3["One-Hot Building Typology"]
        F3 --> F4["Z-Score Feature Standardization"]
    end

    subgraph S3 ["3. Source Pre-Training"]
        direction TB
        M1["Source Domain: Zone 2B (Tempe)"] --> M2["Deep ANN: 256-128-64-32-1"]
        M2 --> M3["5-Fold Cross Validation"]
        M3 --> M4["Adam Optimizer (lr = 1e-5)"]
    end

    subgraph S4 ["4. Climate Transfer & Adaptation"]
        direction TB
        T1["Freeze Layers 1-3: General Physics"] --> T2["Retrain Layer 4 & Output Head"]
        T2 --> T3["Target Domain: Zone 6A (Cornell)"]
        T3 --> T4["Data Fractions: 5% to 100%"]
    end

    S1 ==> S2 ==> S3 ==> S4

    classDef group fill:#F8F9FA,stroke:#1A73E8,stroke-width:2px,font-weight:bold;
    classDef default fill:#FFFFFF,stroke:#5F6368,stroke-width:1.2px;
    class S1,S2,S3,S4 group;
```

---

## 2. Artificial Neural Network Architecture & Parameter Freezing (Figure 2)

```mermaid
flowchart TB
    IN["Input Layer: 16 Features<br>(Meteorological, Cyclic Time, Building Metadata)"]
    
    subgraph FROZEN ["Generalized Physics Layers (Frozen in TL)"]
        direction TB
        L1["Dense Layer 1: 256 Units (ReLU)"]
        L2["Dense Layer 2: 128 Units (ReLU)"]
        L3["Dense Layer 3: 64 Units (ReLU)"]
        L1 --> L2 --> L3
    end

    subgraph TRAIN ["Target-Specific Adaptation (Trainable)"]
        direction TB
        L4["Dense Layer 4: 32 Units (ReLU)"]
        OUT["Output Layer: 1 Unit (Linear)"]
        L4 --> OUT
    end

    PRED["Predicted Energy: expm1(y) Inversion (kWh)"]

    IN --> L1
    L3 --> L4
    OUT --> PRED

    classDef frozen fill:#E8F0FE,stroke:#1A73E8,stroke-width:2px,color:#174EA6,font-weight:bold;
    classDef tuned fill:#E6F4EA,stroke:#137333,stroke-width:2px,color:#0D652D,font-weight:bold;
    classDef io fill:#F1F3F4,stroke:#5F6368,stroke-width:1.5px,color:#202124,font-weight:bold;
    class L1,L2,L3 frozen;
    class L4,OUT tuned;
    class IN,PRED io;
```

