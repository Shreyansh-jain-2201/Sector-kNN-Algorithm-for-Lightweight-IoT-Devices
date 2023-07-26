# Sector-kNN-Algorithm-for-Lightweight-IoT-Devices
The "Sector kNN Algorithm for Lightweight IoT Devices" repository introduces the "Federated Sector kNN" algorithm, tailored for lightweight Internet of Things (IoT) devices. It optimizes prediction time to O(1) and employs transfer learning for improved accuracy. With federated learning, it enables collaborative model training while preserving data privacy. The algorithm demonstrates comparable accuracy to traditional kNN methods on various datasets and supports both classification and regression tasks. It aims to revolutionize machine learning on IoT devices, ensuring efficient and privacy-conscious learning.


![Python](https://img.shields.io/badge/3.6-blue) ![Python](https://img.shields.io/badge/3.7-blue) ![Python](https://img.shields.io/badge/3.8-blue) ![Python](https://img.shields.io/badge/3.9-blue) ![Python](https://img.shields.io/badge/3.10-blue) ![Python](https://img.shields.io/badge/3.11-blue)

![License](https://img.shields.io/badge/license-MIT-green)

## Introduction

The "Sector kNN" algorithm is a novel approach designed to address the computational efficiency and memory usage issues of the K-nearest neighbor (kNN) algorithm, particularly for lightweight Internet of Things (IoT) devices. This implementation aims to provide accurate and efficient predictions for real-time use cases on resource-constrained IoT devices with limited computational power and memory.

### Key Features

- Reduced prediction time complexity from O(N) to O(1).
- Transfer learning tailored for lightweight IoT devices to enhance accuracy.
- Federated learning allows multiple parties to collaboratively train a model without sharing their data. This ensures privacy in sensitive applications like healthcare, where a large dataset can be used for model training without sharing individual data.
- Evaluation on various datasets with comparable accuracy to kd tree kNN and classical kNN.
- Support for both classification and regression tasks.

# Federated Learning model
![FEDERATED LEARNING MODEL](https://github.com/Shreyansh-jain-2201/Sector-kNN-Algorithm-for-Lightweight-IoT-Devices/assets/79896658/8f2e19a3-0d35-4e2b-9890-a4ad745bee9a)

## Federated Learning Workflow

1. **Data Partitioning and Sector Creation:**

   - The feature space is partitioned into sectors using sector-based distance metrics.
   - Each IoT device is responsible for processing and storing data locally within its assigned sector.
2. **Local Model Training:**

   - On each IoT device, a local kNN model is trained using its own data that belongs to its sector.
   - The local model computes the distance between each data point and its neighbors within the same sector.
   - Local models are updated based on the training data and sector-specific information.
3. **Model Aggregation:**

   - After local training, the local models from all IoT devices are aggregated to form a global model.
   - Model aggregation is performed without exchanging raw data between devices, ensuring data privacy.
   - Aggregation may involve combining the results of each local model to create a more robust and accurate global model.
4. **Transfer Learning:**

   - Transfer learning is incorporated during the training process to enhance accuracy.
   - The global model benefits from knowledge learned in the source tasks, which can be useful for the target tasks.
5. **Constant Prediction Time:**

   - The Sector kNN algorithm achieves constant prediction time, regardless of the dataset size or model complexity.
   - This ensures real-time predictions on lightweight IoT devices, even with limited computational resources.
6. **Model Deployment:**

   - The final global model, which is the result of federated training, is deployed back to the IoT devices.
   - Each IoT device can use the global model locally to perform predictions on new data without sharing it with others.

The workflow in the Sector kNN algorithm with Federated Learning allows multiple IoT devices to collaboratively train a shared kNN model while preserving data privacy and decentralization. By leveraging federated learning, the algorithm addresses scalability and privacy concerns, making it suitable for real-world applications in IoT and privacy-sensitive domains like healthcare and finance.

![Federated Learning Workflow](https://github.com/Shreyansh-jain-2201/Sector-kNN-Algorithm-for-Lightweight-IoT-Devices/assets/79896658/086bad0d-4d36-4d31-a071-5a0bec964c1b)

## Usage and Installation

To use the Sector kNN algorithm, follow these steps:

1. Clone this repository:

```bash
git clone https://github.com/Shreyansh-jain-2201/Sector-kNN-Algorithm-for-Lightweight-IoT-Devices.git
cd Sector-kNN-Algorithm-for-Lightweight-IoT-Devices
```

2. Install the required dependencies using `pip`:

```bash
pip install -r requirements.txt
```

3. Import the kNN class from the kNN.py file into your Python code:

```
from kNN import *
```

4. Load the dataset and prepare the features and labels. Ensure that the class labels are encoded as integers (0, 1, 2, ...) and the features are encoded as floating-point numbers.
5. Normalize your features (optional) using either min-max normalization or z-score normalization for better performance.
6. Split your dataset into training and testing sets:

```
from sklearn.model_selection import train_test_split

# Assuming you have 'X' as features and 'y' as labels
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

7. Initialize and train the Sector kNN model:

```
HYPERPARAMETERS__ = [4, 4, 4, 4]  # Define your hyperparameters here

model = kNN()
model.fit(x_train, y_train, HYPERPARAMETERS__, classes=3, k=1, Normalized=True)
model.compile()
```

8. Evaluate the accuracy of the trained model on the test data:

```
model.accuracy(x_test, y_test)
```

9. Optionally, save the trained model to a file:

```
def saveModel(model, name):
    with open(name, 'wb') as file:
        pickle.dump(model, file)

saveModel(model, 'model.pkl')
```

10. To load the saved model, use the following code:

```
def loadModel(name):
    with open(name, 'rb') as file:
        return pickle.load(file)

model__ = loadModel('model.pkl')
```

### Performance

#### Constant prediction time as seen on various datasets:

## [Iris Dataset](https://github.com/Shreyansh-jain-2201/Sector-kNN-Algorithm-for-Lightweight-IoT-Devices/blob/ce68f6a7c05baec9783bad82bf06559a7f590628/Datasets/Iris.csv)

| No. of Sectors | Prediction Accuracy | Model Size (MB) | Training Time (s) | Prediction Time (s) | Learning Time (s) |
| -------------- | ------------------- | --------------- | ----------------- | ------------------- | ----------------- |
| 1              | 33.33%              | 0.0047          | 0.0086            | 0.00150             | 0.001             |
| 4              | 82.66%              | 0.0050          | 0.0048            | 0.00050             | 0.001             |
| 16             | 85.33%              | 0.0065          | 0.0097            | 0.00048             | 0.001             |
| 64             | 88.00%              | 0.0124          | 0.0116            | 0.00102             | 0.001             |
| 256            | 93.33%              | 0.0360          | 0.0356            | 0.00103             | 0.002             |
| 1024           | 96.66%              | 0.1305          | 0.8023            | 0.00079             | 0.001             |
| 4096           | 98.66%              | 0.5085          | 20.4194           | 0.00237             | 0.001             |
| 16384          | 98.0%               | 2.0206          | 375.5257          | 0.0010              | 0.001             |

## [Shuttle Dataset](https://github.com/Shreyansh-jain-2201/Sector-kNN-Algorithm-for-Lightweight-IoT-Devices/blob/ce68f6a7c05baec9783bad82bf06559a7f590628/Datasets/shuttle.csv)

| No. of Sectors | Prediction Accuracy | Model Size (MB) | Training Time (s) | Prediction Time (s) | Learning Time (s) |
| -------------- | ------------------- | --------------- | ----------------- | ------------------- | ----------------- |
| 1              | 79.06%              | 0.8416          | 0.0311            | 0.0015              | 0.001             |
| 4              | 79.17%              | 0.8422          | 0.0283            | 0.0009              | 0.001             |
| 16             | 79.21%              | 0.8447          | 0.02310           | 0.0012              | 0.002             |
| 64             | 87.96%              | 0.8548          | 0.02779           | 0.0008              | 0.001             |
| 256            | 89.86%              | 0.8591          | 0.03643           | 0.0014              | 0.001             |
| 1024           | 91.79%              | 1.0563          | 0.60432           | 0.0012              | 0.001             |
| 4096           | 91.88%              | 1.7009          | 6.7168            | 0.0011              | 0.002             |
| 16384          | 94.57%              | 4.2794          | 45.0785           | 0.0003              | 0.0002            |

## [HTRU Dataset](https://github.com/Shreyansh-jain-2201/Sector-kNN-Algorithm-for-Lightweight-IoT-Devices/blob/ce68f6a7c05baec9783bad82bf06559a7f590628/Datasets/HTRU_2.csv)

| No. of Sectors | Prediction Accuracy | Model Size (MB) | Training Time (s) | Prediction Time (s) | Learning Time (s) |
| -------------- | ------------------- | --------------- | ----------------- | ------------------- | ----------------- |
| 1              | 90.84%              | 0.8200          | 0.0319            | 0.0006              | 0.001             |
| 4              | 93.06%              | 0.8205          | 0.0263            | 0.0008              | 0.001             |
| 16             | 93.91%              | 0.8224          | 0.0275            | 0.0010              | 0.001             |
| 64             | 97.20%              | 0.8302          | 0.0294            | 0.0010              | 0.001             |
| 256            | 97.51%              | 0.8613          | 0.0414            | 0.0005              | 0.001             |
| 1024           | 97.72%              | 0.9859          | 0.2608            | 0.0007              | 0.002             |
| 4096           | 97.76%              | 1.4840          | 4.6422            | 0.0008              | 0.001             |
| 16384          | 97.85%              | 3.4765          | 66.1641           | 0.0004              | 0.0001            |

## [Poker Dataset](https://github.com/Shreyansh-jain-2201/Sector-kNN-Algorithm-for-Lightweight-IoT-Devices/blob/ce68f6a7c05baec9783bad82bf06559a7f590628/Datasets/poker.csv)

| No. of Sectors | Prediction Accuracy | Model Size (MB) | Training Time (s) | Prediction Time (s) | Learning Time (s) |
| -------------- | ------------------- | --------------- | ----------------- | ------------------- | ----------------- |
| 1              | 50.12%              | 65.6906         | 1.4816            | 0.0009              | 0.00009           |
| 4              | 92.36%              | 65.6913         | 1.4837            | 0.0015              | 0.00009           |
| 16             | 92.36%              | 65.6942         | 1.4616            | 0.0010              | 0.0001            |
| 64             | 92.36%              | 65.7055         | 1.4648            | 0.0010              | 0.00009           |
| 256            | 92.36%              | 65.6554         | 1.4149            | 0.0005              | 0.001             |
| 1024           | 92.72%              | 65.5503         | 1.6649            | 0.0005              | 0.002             |
| 4096           | 92.76%              | 66.6554         | 1.7890            | 0.0005              | 0.001             |
| 16384          | 92.76%              | 69.5503         | 2.3070            | 0.0005              | 0.001             |

## [WiFi Dataset](https://github.com/Shreyansh-jain-2201/Sector-kNN-Algorithm-for-Lightweight-IoT-Devices/blob/ce68f6a7c05baec9783bad82bf06559a7f590628/Datasets/wifi.csv)

| No. of Sectors | Prediction Accuracy | Model Size (MB) | Training Time (s) | Prediction Time (s) | Learning Time (s) |
| -------------- | ------------------- | --------------- | ----------------- | ------------------- | ----------------- |
| 1              | 25.0%               | 0.0923          | 0.0049            | 0.0002              | 0.0001            |
| 4              | 79.05%              | 0.0928          | 0.0040            | 0.0010              | 0.00009           |
| 16             | 97.3%               | 0.0949          | 0.0040            | 0.0003              | 0.00009           |
| 64             | 97.6%               | 0.1030          | 0.0050            | 0.0009              | 0.0001            |
| 256            | 97.95%              | 0.1356          | 0.0099            | 0.0002              | 0.00009           |
| 1024           | 98.05%              | 0.2660          | 0.0615            | 0.0003              | 0.0001            |
| 4096           | 98.35%              | 0.7876          | 1.2837            | 0.0010              | 0.00009           |
| 16384          | 99.3%               | 2.8738          | 26.7988           | 0.0010              | 0.0001            |

## Visualization and Interpretability

The Sector kNN algorithm provides utility functions for visualizing the clustered data points and generating a confusion matrix for evaluating the classification performance. Additionally, the integration of SHAP (SHapley Additive exPlanations) allows for understanding the model's predictions and feature contributions.

```
# Visualization of clustered data points
model.plotClusters(x_test, y_test)

# Confusion matrix chart
model.plotConfusionChart(x_test, y_test)

# Classification report
model.showClassificationReport(x_test, y_test)
```

## Contributing

Contributions to this project are welcome! If you have any suggestions, bug fixes, or feature enhancements, please feel free to open an issue or submit a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
