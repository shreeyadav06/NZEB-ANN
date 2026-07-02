# **Important Research Findings**

## (To be included in the manuscript’s relevant sections)

![][image1]

# **Dataset Selection and Explorations**

### **Anonymized site\_id issue in ASHRAE**

The ASHRAE dataset is anonymized and does not include a specific column for city names; instead, it uses a **site\_id** ranging from 0 to 15\. To map our anonymized site IDs, we had to analyze the weather parameters provided in the `weather_train.csv` (such as air temperature and humidity) and match them against the five main climate zones defined in the literary sources:

* **Zone A (Equatorial):** High temperatures and significant precipitation year-round.  
* **Zone B (Arid):** Low rainfall and high solar radiation; these regions typically require heavy active cooling systems.  
* **Zone C (Warm-temperate):** Moderate temperatures, like the oceanic climate of **Cébazat, France** (Csb).  
* **Zone D (Snow):** Harsh winters requiring effective heating, such as the subarctic climate of **Chita, Russia** (Dwc) or **Lhasa, China**.  
* **Zone E (Polar):** Extremely cold regions with minimal solar gain in winter, exemplified by **Resolute, Canada** (ET)

### **Have any other literatures used anything of these 2 datasets?**

None of the primarily picked prior studies utilized the **ASHRAE Great Energy Predictor** dataset or the **Building Data Genome Project 2 (bdgp2)** dataset for training their machine learning models. Instead, these sources exclusively reference **ASHRAE Standards** as regulatory guidelines for design compliance and the mathematical formulation of Net-Zero Energy Building (NZEB) constraints. 

“Rather than using pre-existing datasets like ASHRAE or bdgp2, the researchers in these sources relied on **simulated or regional datasets** tailored to their specific study areas. **Ibrahim et al.** and **Farahmandfar et al.** both generated their own high-fidelity datasets using the **TRNSYS (Transient System Simulation Tool)** environment, with the former producing 1.752 million hourly records through **Latin Hypercube Sampling** to ensure a well-distributed training set,,. Furthermore, **Alghassab** utilized regional historical operational data and environmental variables specific to **Riyadh, Saudi Arabia, and Lhasa, China**, rather than using the anonymized site data typically found in the ASHRAE Great Energy Predictor,. While these papers demonstrate that the ASHRAE standards are central to the definitions used in NZEB literature, they prioritize site-specific or simulation-generated data over the mentioned open-source benchmark datasets. “

### **Should we use pre-made datasets, or generate/synthesize our own?**

While many of the current papers, such as those by **Ibrahim et al.** and **Farahmandfar et al.**, rely on data generated via **TRNSYS** or **EnergyPlus**, they explicitly identify the high computational cost and time required for these simulations—often taking days just for data generation. By using the ASHRAE or BGDP2 dataset, our project addresses the literature's call for **scalable, data-driven solutions** that can bypass these slow physics-based bottlenecks. Using a high-quality, verified real-world benchmark dataset demonstrates that your model can handle **complex, non-linear operational data**, which is a "next-generation" step beyond the idealized environments of simulations 

A primary gap identified in our literature survey is that most studies are **climate-locked**, focusing on only one or two cities (e.g., Riyadh and Lhasa in the Alghassab study). A research paper that utilizes a **16-site benchmark** like ASHRAE is significantly more robust because it tests **generalizability** across a far wider range of building types and climatic zones. The latest research specifically notes that "exploring transfer learning" is a key future direction to improve model generalization. By being among the first to apply transfer learning specifically to this large-scale NZEB context, we are directly **filling a distinct and noticeable gap** highlighted in the most recent 2026 sources. Further, In AI/ML research, publishing results on a recognized **benchmark dataset** is often viewed as more rigorous than using a private, self-generated dataset. It allows other researchers to **reproduce and verify your results**.

### **How we De-Anonymized the site\_ids in the ASHRAE dataset?**

While the official competition data left these locations completely hidden to prevent contestants from fetching external weather data, data scientists successfully de-anonymised them. They matched the distinct patterns of the dataset's target variables, timezones, and hourly weather signatures (temperature, dew point, and pressure) to precise meteorological and municipal data. \[[1](https://medium.com/analytics-vidhya/ashrae-great-energy-predictor-iii-a-machine-learning-case-study-a01a67eb048d), [2](https://www.kaggle.com/code/patrick0302/locate-cities-according-weather-temperature)\]

The 16 site\_ids mappings represent the following real-world locations:

| site\_id \[[1](https://www.sciencedirect.com/science/article/pii/S1569843222002965)\] | True Geographic Location / Entity | Country |
| ----- | ----- | ----- |
| **0** | University of Central Florida (Orlando, FL) | United States |
| **1** | University College London (London) | United Kingdom |
| **2** | Arizona State University (Tempe, AZ) | United States |
| **3** | Washington D.C. area metro buildings | United States |
| **4** | University of California, Berkeley (Berkeley, CA) | United States |
| **5** | United Kingdom site (De-anonymised as a UK campus) | United Kingdom |
| **6** | Philadelphia, Pennsylvania | United States |
| **7** | Montreal, Quebec | Canada |
| **8** | Orlando, Florida (Alternative municipality/cluster) | United States |
| **9** | University of Texas at Austin (Austin, TX) | United States |
| **10** | Salt Lake City, Utah | United States |
| **11** | Ottawa, Ontario | Canada |
| **12** | University of Minnesota (Minneapolis, MN) | United States |
| **13** | Cambridge, Massachusetts (Likely MIT/Harvard area) | United States |
| **14** | Coastal Central California area | United States |
| **15** | Cornell University (Ithaca, NY) | United States |

---

How We Know the Locations (Methodology)

Data scientists discovered these exact coordinates and identities during and immediately after the competition through a combination of **weather alignment**, **metadata footprints**, and **leakage exploration**:

1. **Weather Timestamp Target Matching:** The auxiliary weather files (weather\_train.csv and weather\_test.csv) contained precise hourly metrics like air temperature and wind speed. Competitors scraped historical weather logs globally from the National Oceanic and Atmospheric Administration (NOAA) database and calculated mathematical correlation coefficients. A near 100% correlation easily pinpointed the exact local weather station for each site.   
2. **Timezone Offset Calculations:** Aligning the "cycles" of building energy consumption spikes (which naturally rise during local business hours) against the raw timestamps revealed the local UTC offset (e.g., Eastern, Mountain, or Greenwich Mean Time), narrowing down continents and coastal zones.   
3. **Building Profile "Leaking" Metadata:** Distinct combinations of building parameters—such as exact square\_feet, the year\_built, and floor\_count—matched public energy disclosure data mandatorily published by major universities and municipalities (such as New York City's Local Law 84 or Seattle's Energy Benchmarking data). 

### **Which sites were chosen for the project?**

The ASHRAE Great Energy Predictor III dataset was used as the primary experimental dataset. The dataset consists of three relational tables: train.csv, weather\_train.csv and building\_metadata.csv. To resolve the anonymized site identifiers into real-world locations and climate zones, metadata published as part of the Building Data Genome Project 2 (BDGP2) was used. According to BDGP2, Site 2 corresponds to Arizona State University (Tempe, Arizona; Climate Zone 2B) and Site 15 corresponds to Cornell University (Ithaca, New York; Climate Zone 6A). These sites are selected as the source and target domains respectively because of their strong climatic contrast. 

### 

### **Possible risk of chosen problem statement?**

Our methodology currently sounds very ambitious, but the contribution may be viewed as **incremental** if we simply train an ANN, transfer weights, freeze layers and compare against scratch training since many papers have already explored transfer learning for energy prediction.

We need a sharper contribution. Possible angles:

* Climate-zone-aware transfer learning.  
* Quantifying transferability between climate classes.  
* Determining which layers should be frozen.  
* Identifying the minimum target-data percentage required.  
* Cross-continental transfer experiments.  
* Explainability analysis of transferred features.

### 

### **Possible pivot/slight deviation?**

Instead of "Transfer learning for building energy prediction", we slightly change it to  make the paper: "Quantifying Climate-Aware Transferability in Building Energy Prediction".

**Proposed hypothesis: “**Transfer learning effectiveness is inversely proportional to climatic dissimilarity between source and target regions.”

**Final contribution statement:** We propose a climate-aware transfer-learning framework for building energy prediction and quantitatively investigate:

* “the effect of climate dissimilarity on transfer performance”,  
* “the minimum target-data requirement”,  
* “and the optimal degree of parameter transfer between climate domains.”

**Proposed Methodology:** 

1. **Source-Target findings:**

| Source | Target |
| ----- | ----- |
| Tempe | Berkeley |
| Tempe | Cornell |
| Berkeley | Cornell |

2. **Construct Transferability Matrix**

| Source | Target | Transfer Gain |
| ----- | ----- | ----- |
| 2B | 3C | \+18% |
| 2B | 6A | \+7% |
| 3C | 6A | \+12% |

   

3. **Discovery question:** Transfer performance decreases as climate distance increases.   
4. **FINAL TITLE:** "Quantifying Climate-Aware Transferability in Building Energy Prediction" 

[image1]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAnAAAAAECAYAAAAOPwJdAAAANElEQVR4Xu3WMQ0AMAwEseAOoJIpqGYvgrzkwcshuLqnHwAAOeoPAADsZuAAAMIYOACAMAOTpGslhjl7rwAAAABJRU5ErkJggg==>