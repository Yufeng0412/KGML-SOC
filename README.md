# KGML-SOC
Develop GRU-based KGML model for simulating annual SOC change and crop yield using synthetic data from ecosys.

# Model structure
The recently published KGMLag model shed lights on the leverage of domain knowledge to improve ML simulations of agroecosystem time-seqeuences.
KGMLag has a hierarchical structure of 4-GRU cells to mimic the carbon and water cycling of crop photosynthesis in the process-based model, ecosys (Liu et al., 2024. DOI: 10.1038/s41467-023-43860-5).

This work was inspired by KGMLag, aiming to build a KGML model for simulating SOC dynamics, by incoporating KGML-ag hidden layers as embedding. SOC is a difficult time-sequence to capture for ML models due to the its subtle variations compared to the magnitude and the combination of a fluctuating seasonal variation and a step-wise annual trend. To deal with this issue, we decide to target at delta_SOC, the annual SOC change in our model development.

There are 3 models in comparison: (1) a baseline GRU model; (2) KGML-ag-SOC1, which takes KGML-ag embedding; (3) KGML-ag-SOC2, which adds two intermediate variables to KGML-ag-SOC1 (domain knowledge).
The model structures are shown as below: 
![KGML-SOC](https://github.com/user-attachments/assets/7883ed5d-16c1-473b-ae68-07c9dc34de66)

The results indicate that KGML-ag-SOC outperformed GRU-SOC model in both delta_SOC and crop yield. Introducing important intermediate variables from the previous year as constraints for the next year helped further imporve the performance, especailly on delta_SOC simulation:

![KGML-SOC_results1](https://github.com/user-attachments/assets/45887966-b895-4263-96f6-41fd74c4c8d0) ![KGML-SOC_results2](https://github.com/user-attachments/assets/be478004-16a2-4b0e-95ac-11c6b3d9b6a5)







