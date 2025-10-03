# rice-classification-cnn
Training and testing some Convolutional Neural Networks using PyTorch for the Deep Learning Course Agust-Dec 2025


---

# Rice Image Dataset (extrated from Kaggle)

## **Overview**
The **Rice Image Dataset** is a collection of high-resolution images designed for rice grain classification. It contains images of five different rice varieties, making it useful for machine learning applications in agricultural research, food quality assessment, and automated classification systems.

## **Dataset Details**
- **Total Images**: 75,000
- **Number of Classes**: 5
- **Image Size**: 250 × 250 pixels
- **Format**: JPEG
- **Grayscale or RGB**: RGB

## **Classes (Rice Varieties)**
The dataset includes images of the following five types of rice grains:
1. **Arborio**
2. **Basmati**
3. **Ipsala**
4. **Jasmine**
5. **Karacadag**

Each class contains an equal number of images, ensuring balanced data for training machine learning models.

## **Potential Use Cases**
- **Classification Models**: Train deep learning models (e.g., CNNs, ViTs) to classify rice varieties.
- **Feature Extraction**: Extract texture, shape, and color features for distinguishing different rice types.
- **Quality Assessment**: Identify high-quality vs. low-quality grains using automated inspection systems.
- **Agricultural Research**: Analyze grain characteristics to improve production and processing techniques.


# Rice Image Dataset
DATASET: https://www.muratkoklu.com/datasets/

Citation Request: See the articles for more detailed information on the data.

Koklu, M., Cinar, I., & Taspinar, Y. S. (2021). Classification of rice varieties with deep learning methods. Computers and Electronics in Agriculture, 187, 106285. https://doi.org/10.1016/j.compag.2021.106285

Cinar, I., & Koklu, M. (2021). Determination of Effective and Specific Physical Features of Rice Varieties by Computer Vision In Exterior Quality Inspection. Selcuk Journal of Agriculture and Food Sciences, 35(3), 229-243. https://doi.org/10.15316/SJAFS.2021.252

Cinar, I., & Koklu, M. (2022). Identification of Rice Varieties Using Machine Learning Algorithms. Journal of Agricultural Sciences https://doi.org/10.15832/ankutbd.862482

Cinar, I., & Koklu, M. (2019). Classification of Rice Varieties Using Artificial Intelligence Methods. International Journal of Intelligent Systems and Applications in Engineering, 7(3), 188-194. https://doi.org/10.18201/ijisae.2019355381

DATASET: https://www.muratkoklu.com/datasets/

*Highlights*

- Arborio, Basmati, Ipsala, Jasmine and Karacadag rice varieties were used.
- The dataset (1) has 75K images including 15K pieces from each rice variety. The dataset (2) has 12 morphological, 4 shape and 90 color features.
- ANN, DNN and CNN models were used to classify rice varieties.
- Classified with an accuracy rate of 100% through the CNN model created.
- The models used achieved successful results in the classification of rice varieties.

 *Abstract*

Rice, which is among the most widely produced grain products worldwide, has many genetic varieties. These varieties are separated from each other due to some of their features. These are usually features such as texture, shape, and color. With these features that distinguish rice varieties, it is possible to classify and evaluate the quality of seeds. In this study, Arborio, Basmati, Ipsala, Jasmine and Karacadag, which are five different varieties of rice often grown in Turkey, were used. A total of 75,000 grain images, 15,000 from each of these varieties, are included in the dataset. A second dataset with 106 features including 12 morphological, 4 shape and 90 color features obtained from these images was used. Models were created by using Artificial Neural Network (ANN) and Deep Neural Network (DNN) algorithms for the feature dataset and by using the Convolutional Neural Network (CNN) algorithm for the image dataset, and classification processes were performed. Statistical results of sensitivity, specificity, prediction, F1 score, accuracy, false positive rate and false negative rate were calculated using the confusion matrix values of the models and the results of each model were given in tables. Classification successes from the models were achieved as 99.87% for ANN, 99.95% for DNN and 100% for CNN. With the results, it is seen that the models used in the study in the classification of rice varieties can be applied successfully in this field.
