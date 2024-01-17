# PyMLDA - Machine Learning for Damage Assessment

### About
The PyMLDA software refers to a computer program based on Python that provides a comprehensive and effective approach to monitoring the structural integrity of systems and structures, using signatures based on dynamics and vibration. By integrating experimental data and machine learning techniques, the software offers an effective tool for assessing structural integrity through the dynamic response of structures. The approach combines supervised and unsupervised machine learning (ML) algorithms to address the challenges associated with assessing damage in a structure.
The applied dataset uses dynamic response (natural frequency, frequency response function (FRF), time or frequency spectrum) obtained from numerical or experimental tests to calculate a damage index (DI) under healthy and damaged conditions, serving as input for the algorithms. The framework employs a variety of validation and cross-validation metrics to assess the effectiveness and accuracy of these machine-learning algorithms in detecting and diagnosing structure-related issues. The PyMLDA software includes graphical features for visualizing the results of the classification model.
Initially, the algorithm aims to automatically extract patterns and features from the data by reading through damage indices. Subsequently, the algorithm is employed to identify the damage state using classification techniques (damaged or healthy) and regression (e.g., damage size, variability associated with estimation, etc.).
From the processed data, if necessary for the user, it is possible to determine actions to correct the structure or components and implement appropriate measures. Hence, PyMLDA software applies to monitoring structures through their dynamic response, using machine learning algorithms to identify and quantify damage.

### ☑️ Prerequisites

Before running the framework, there is a set of libraries to be installed:

    - numpy >= 1.26.1
    - pandas >= 2.1.3
    - Pandas >= 0.25.2
    - scikit_learn >= 1.3.2
    - scipy >= 1.11.3



### PyMLDA workflow

The software encompasses eight steps in total, comprising receiving the normalised acquired data (step 1), an unsupervised stage involving data processing (step 2), feature selection (step 3), and pattern recognition and clustering (step 4). These steps form the Data-Driven Processing and Pattern Recognition. Subsequently, data splitting is performed in step 5. In the supervised stage, classification ML algorithms (steps 6 and 7) are utilised for damage detection, and regression (steps 6 and 7) is applied for damage quantification. Finally (step 8), the algorithm furnishes information regarding the damage state based on the classification and regression algorithm outcomes. The workflow of the steps is illustrated in the following figure. 
<p align="center">
  <img src="ProcessML_PR_SHM.png" width="85%">
</p>


<!--## 🚀 Using the Framework

Do you want to start working with **CAPRI**? It is pretty easy! Just clone the repository and follow the instructions below:

> ⏳ The framework works on **Python 3.9.x** for now, and will be upgraded to newer versions in the recent future.

> ⏳ We are working on making the repository available on **pip** package manager. Hence, in the next versions, you will not need to clone the framework anymore.-->



### 🚀 Launch the PyMLDA

You can start the project by running the `PyMLDA_Machine_Learning_for_Damage_Assessment.ipynb` or `pymlda_machine_learning_for_damage_assessment.py` file in the root directory. With this, the application settings are loaded from the DI dataset `DI_FRAC_Exp-estimation.xlxs`, which is also available in the repository and described in [1,2]. The system starts processing the DI dataset using the selected model and provides some evaluations on it. The final output is a classification of the system's healthy condition as damaged or healthy and the damage quantification with a variability associated with the estimation.

<!--```python
# The evaluation file containing the selected evaluation metrics - It shows that the user selected GeoSoCa model on Gowalla dataset with Product fusion type, applied on 5628 users where the top-10 results are selected for evaluation and the length of the recommendation lists are 15
Eval_GeoSoCa_Gowalla_Product_5628user_top10_limit15.csv
# The txt file containing the evaluation lists with the configurations described above
Rec_GeoSoCa_Gowalla_Product_5628user_top10_limit15.txt 
```-->

## 🧩 Contribution Guide

Contributing to open-source codes is a rewarding method to learn, teach, and gain experience. We welcome all contributions, from bug fixes to new features and extensions. Do you want to be a contributor to the project? Read more about it on our page (https://sites.google.com/view/marcelamachado/publications/open-source).

<!-- ## Team

CAPRI is developed with ❤️ by:

| <a href="https://github.com/alitourani"><img src="https://github.com/alitourani.png?size=70"></a> | <a href="https://github.com/rahmanidashti"><img src="https://github.com/rahmanidashti.png?size=70"></a> | <a href="https://github.com/naghiaei"><img src="https://github.com/naghiaei.png" width="70"></a> | <a href="https://github.com/yasdel"><img src="https://yasdel.github.io/images/yashar_avator.jpg" width="70"></a> |
| ------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------- |
| [Ali Tourani](mailto:ali.tourani@uni.lu "ali.tourani@uni.lu")                                     | [Hossein A. Rahmani](mailto:h.rahmani@ucl.ac.uk "h.rahmani@ucl.ac.uk")                                  | [MohammadMehdi Naghiaei](mailto:naghiaei@usc.edu "naghiaei@usc.edu")                             | [Yashar Deldjoo](mailto:yashar.deldjoo@poliba.it "yashar.deldjoo@poliba.it")                                     | -->


## 📝 Citation

If you find **PyMLDA** useful for your research or development, please cite the  following:

```
@inproceedings{PyMLDA2024,
  title={PyMLDA - Machine Learning for Damage Assessment},
  author={ Coelho, J.S. and Machado, M.R. and  Sousa, A.A.S.R.},
  booktitle={},
  year={2024}
}
```

## 🟢 Versions
- Version 1
  - Implementation of the PyMLDA with a simple GUI
  - Supporting as input only the damage index already processed. 


## 🟠 Next realease

- Incorporation of Damage Index calculation and selection of the better DI
- Raw data will be given as input (Natural Frequency, temporal and frequency response of the dynamic system)


##  References

[1] Amanda A.S.R. de Sousa, Marcela R. Machado, Experimental vibration dataset collected of a beam reinforced with masses under different health conditions, Data in Brief, 2024, 110043,ISSN 2352-3409,
https://doi.org/10.1016/j.dib.2024.110043.

[2] A. A. S. R. D. Sousa, and M. R. Machado. “Damage Assessment of a Physical Beam Reinforced with Masses - Dataset”. Multiclass Supervised Machine Learning Algorithms Applied to Damage and Assessment Using Beam Dynamic Response. Zenodo, November 8, 2023. https://doi.org/10.5281/zenodo.8081690.

[3] Coelho, J.S., Machado, M.R., Dutkiewicz, M. et al. Data-driven machine learning for pattern recognition and detection of loosening torque in bolted joints. J Braz. Soc. Mech. Sci. Eng. 46, 75 (2024). https://doi.org/10.1007/s40430-023-04628-6
