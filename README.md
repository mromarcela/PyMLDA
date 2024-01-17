# PyMLDA - Machine Learning for Damage Assessment
**General Information:** The PyMLDA software refers to a computer program based on Python that provides a comprehensive and effective approach to monitoring the structural integrity of systems and structures, using signatures based on dynamics and vibration. By integrating experimental data and machine learning techniques, the software offers an effective tool for assessing structural integrity through the dynamic response of structures. The approach combines supervised and unsupervised machine learning (ML) algorithms to address the challenges associated with assessing damage in a structure.
The applied dataset uses dynamic response (natural frequency, frequency response function (FRF), time or frequency spectrum) obtained from numerical or experimental tests to calculate a damage index (DI) under healthy and damaged conditions, serving as input for the algorithms. The framework employs a variety of validation and cross-validation metrics to assess the effectiveness and accuracy of these machine-learning algorithms in detecting and diagnosing structure-related issues. The PyMLDA software includes graphical features for visualizing the results of the classification model.
Initially, the algorithm aims to automatically extract patterns and features from the data by reading through damage indices. Subsequently, the algorithm is employed to identify the damage state using classification techniques (damaged or healthy) and regression (e.g., damage size, variability associated with estimation, etc.).
From the processed data, if necessary for the user, it is possible to determine actions to correct the structure or components and implement appropriate measures. Hence, PyMLDA software applies to monitoring structures through their dynamic response, using machine learning algorithms to identify and quantify damage.

**How PyMLDA works**

The software encompasses eight steps in total, comprising receiving the normalised acquired data (step 1), an unsupervised stage involving data processing (step 2), feature selection (step 3), and pattern recognition and clustering (step 4). These steps form the Data-Driven Processing and Pattern Recognition. Subsequently, data splitting is performed in step 5. In the supervised stage, classification ML algorithms (steps 6 and 7) are utilised for damage detection, and regression (steps 6 and 7) is applied for damage quantification. Finally (step 8), the algorithm furnishes information regarding the damage state based on the classification and regression algorithm outcomes. The workflow of the steps is illustrated in the following figure. 
<p align="center">
  <img src="ProcessML_PR_SHM.png" width="85%">
</p>


