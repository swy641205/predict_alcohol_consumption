import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


class SVMTrain:
    def __init__(self, df, target):
        self.target = target
        self.df = df
        self.model = None
        self.scaler = StandardScaler()
        self.best_params = None
        self.report = None
        self.accuracy = None

    def mymodel(self):
        X = self.df[['sex', 'guardian', 'Fjob', 'Fedu', 'Medu', 'studytime', 'famrel', 'goout', 'health', 'absences']]
        y = self.df[self.target]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)

        param_grid = {
            'C': [0.1, 1, 10, 100],
            'gamma': [1, 0.1, 0.01, 0.001],
            'kernel': ['rbf', 'poly', 'sigmoid']
        }

        grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=2, cv=5)
        grid.fit(X_train, y_train)

        self.best_params = grid.best_params_
        self.model = grid
        y_pred = grid.predict(X_test)

        self.report = classification_report(y_test, y_pred, output_dict=True)
        self.accuracy = accuracy_score(y_test, y_pred)

        print(f"---------- Best params: {self.best_params}")
        print(f"---------- Classification report: {self.report}")
        result = {
            'best_params': self.best_params,
            'classification_report': classification_report(y_test, y_pred),
            'accuracy': self.accuracy
        }

        return y_test, y_pred, result

    def confusion_matrix(self, y_test, y_pred):
        conf_matrix = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(10, 7))
        sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=range(1, 6), yticklabels=range(1, 6))
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        return plt

        
    def run(self):
        y_test, y_pred, result= self.mymodel()        
        img = self.confusion_matrix(y_test, y_pred)

        return result, img


class SVMPredict:
    def __init__(self, svm, input_data):
        self.svm = svm
        self.df = svm.df
        self.input_data = input_data
        
    def predict_new_data(self):
        new_df = pd.DataFrame(self.input_data)
        new_scaled = self.svm.scaler.transform(new_df)
        predictions = self.svm.model.predict(new_scaled)
        print("Predictions :", predictions)
        return predictions
        
    def plot_comparison(self, predictions):
        sex_values = self.df['sex']
        guardian_values = self.df['guardian']
        fjob_values = self.df['Fjob']
        fedu_values = self.df['Fedu']
        medu_values = self.df['Medu']
        studytime_values = self.df['studytime']
        famrel_values = self.df['famrel']
        goout_values = self.df['goout']
        health_values = self.df['health']
        absences_values = self.df['absences']
        dalc_values = self.df['Dalc']
        walc_values = self.df['Walc']

        
        fig, axs = plt.subplots(4, 3, figsize=(15, 15))

        
        axs[0, 0].hist(sex_values, bins=3, alpha=0.7, label='Original Data')
        axs[0, 0].axvline(self.input_data['sex'][0], color='r', linestyle='dashed', linewidth=2, label='You')
        axs[0, 0].set_title('Sex')

        axs[0, 1].hist(guardian_values, bins=3, alpha=0.7, label='Original Data')
        axs[0, 1].axvline(self.input_data['guardian'][0], color='r', linestyle='dashed', linewidth=2, label='You')
        axs[0, 1].set_title('Guardian')

        axs[0, 2].hist(fjob_values, bins=5, alpha=0.7, label='Original Data')
        axs[0, 2].axvline(self.input_data['Fjob'][0], color='r', linestyle='dashed', linewidth=2, label='You')
        axs[0, 2].set_title('Fjob')

        axs[1, 0].hist(fedu_values, bins=5, alpha=0.7, label='Original Data')
        axs[1, 0].axvline(self.input_data['Fedu'][0], color='r', linestyle='dashed', linewidth=2, label='You')
        axs[1, 0].set_title('Fedu')

        axs[1, 1].hist(medu_values, bins=5, alpha=0.7, label='Original Data')
        axs[1, 1].axvline(self.input_data['Medu'][0], color='r', linestyle='dashed', linewidth=2, label='You')
        axs[1, 1].set_title('Medu')

        axs[1, 2].hist(studytime_values, bins=4, alpha=0.7, label='Original Data')
        axs[1, 2].axvline(self.input_data['studytime'][0], color='r', linestyle='dashed', linewidth=2, label='You')
        axs[1, 2].set_title('Studytime')

        axs[2, 0].hist(famrel_values, bins=5, alpha=0.7, label='Original Data')
        axs[2, 0].axvline(self.input_data['famrel'][0], color='r', linestyle='dashed', linewidth=2, label='You')
        axs[2, 0].set_title('Famrel')
        
        axs[2, 1].hist(goout_values, bins=5, alpha=0.7, label='Original Data')
        axs[2, 1].axvline(self.input_data['goout'][0], color='r', linestyle='dashed', linewidth=2, label='You')
        axs[2, 1].set_title('Goout')

        axs[2, 2].hist(health_values, bins=5, alpha=0.7, label='Original Data')
        axs[2, 2].axvline(self.input_data['health'][0], color='r', linestyle='dashed', linewidth=2, label='You')
        axs[2, 2].set_title('Health')

        axs[3, 0].hist(absences_values, bins=10, alpha=0.7, label='Original Data')
        axs[3, 0].axvline(self.input_data['absences'][0], color='r', linestyle='dashed', linewidth=2, label='You')
        axs[3, 0].set_title('Absences')

        axs[3, 1].hist(walc_values, bins=5, alpha=0.7, label='Original Data')
        axs[3, 1].axvline(predictions[0], color='r', linestyle='dashed', linewidth=2, label='You')
        axs[3, 1].set_title('predictions')

        for ax in axs.flat:
            ax.legend()
        fig.delaxes(axs[3, 2])
        plt.tight_layout()
        plt.show()