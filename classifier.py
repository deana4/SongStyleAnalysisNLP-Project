import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score


class Classifier:
    def __init__(self, csv_file, batchSize=48):
        self.data = pd.read_csv(csv_file)
        groups = self.data.groupby('MappedStyle')

        labels = [0, 1, 9, 18]

        self.dataList = []
        for label, group in groups:
            if label in labels:
                # Create batches and calculate average for each batch
                num_batches = int(np.ceil(len(group) / batchSize))
                batches = np.array_split(group, num_batches)
                batch_list = []

                for batch in batches:
                    batchSum = batch.mean(numeric_only=True).to_frame().T
                    batchSum['MappedStyle'] = label
                    batch_list.append(batchSum)

                startDF = pd.concat(batch_list, ignore_index=True)
                self.dataList.append(startDF)

        self.data = pd.concat(self.dataList)

        self.X = self.data.iloc[:, :-1]
        self.y = self.data.iloc[:, -1]
        self.models = {
            "RandomForest": RandomForestClassifier(),
            "LogisticRegression": LogisticRegression(max_iter=1000),
            "SVM (Linear)": SVC(kernel='linear'),
            "SVM (RBF)": SVC(kernel='rbf'),
            "NeuralNetwork": MLPClassifier(max_iter=50_000)
        }
        self.normalize_data()

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.3,
                                                                                random_state=42)

    def normalize_data(self):
        scaler = MinMaxScaler()
        # scaler = StandardScaler()

        self.X = scaler.fit_transform(self.X)

    def select_top_features(self):
        selector = SelectKBest(score_func=f_classif, k=3)
        selector.fit(self.X_train, self.y_train)
        top_features = selector.get_support(indices=True)
        return self.data.columns[0:-1][top_features]

    def run(self):
        print(self.select_top_features())
        for model_name, model in self.models.items():
            print(f'current model: {model_name}')
            model.fit(self.X_train, self.y_train)

            train_predictions = model.predict(self.X_train)
            test_predictions = model.predict(self.X_test)

            train_accuracy = accuracy_score(self.y_train, train_predictions)
            test_accuracy = accuracy_score(self.y_test, test_predictions)

            print(f"\nModel: {model_name}")
            print(f"Train Accuracy: {train_accuracy:.4f}")
            print(f"Test Accuracy: {test_accuracy:.4f}")


if __name__ == "__main__":
    classifier_comparison = Classifier("fullNumericalDataset.csv")
    classifier_comparison.run()
