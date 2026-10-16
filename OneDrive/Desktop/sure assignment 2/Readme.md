# Customer Segmentation Using K-Means Clustering

## Project Overview
This project uses Unsupervised Machine Learning to segment mall customers into different groups based on their Annual Income and Spending Score.

The K-Means Clustering algorithm is used to identify customers with similar spending behaviors.

## Dataset
**Source:** [Customer Segmentation Tutorial in Python - Kaggle](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python)

Dataset: `Mall_Customers.csv`

Important columns:
- `CustomerID`: Unique customer ID
- `Gender`: Customer gender
- `Age`: Customer age
- `Annual Income (k$)`: Annual income
- `Spending Score (1-100)`: Spending behavior score

## Technologies Used
- Python
- Pandas
- Matplotlib
- Scikit-learn
- K-Means Clustering

## Project Workflow
1. Load the dataset using Pandas.
2. Explore the dataset and check for missing values.
3. Select Annual Income and Spending Score as features.
4. Use the Elbow Method to help choose the number of clusters.
5. Train the K-Means clustering model.
6. Assign cluster labels to customers.
7. Visualize customer segments and cluster centroids.
8. Export the clustered dataset to a CSV file.

## Installation

Install the required libraries:

```bash
pip install pandas matplotlib scikit-learn
```

## How to Run

1. Download the dataset from Kaggle.
2. Place `Mall_Customers.csv` in the project folder.
3. Run the Python script:

```bash
python main.py
```

## Output
- Elbow Method graph
- Customer clustering visualization
- Cluster labels assigned to customers
- `Customer_Segments.csv` containing customer data and cluster labels

## Algorithm
**K-Means Clustering** groups data points into clusters by minimizing the within-cluster sum of squared distances.

This project uses five clusters as an initial setting. The Elbow Method can help evaluate whether this is a suitable choice.

## Learning Outcomes
- Understanding Unsupervised Machine Learning
- Learning K-Means Clustering
- Applying the Elbow Method
- Visualizing customer segments
- Saving model-generated cluster labels

## Author
Palak Verma