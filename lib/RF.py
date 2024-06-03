import pandas as pd
from sklearn.ensemble import RandomForestClassifier


class RandomForest:
    def __init__(self, df: pd.DataFrame, target: str, features: list):
        self.df = df
        self.target = target
        self.features = features
        self.model = self.train_model()

    def train_model(self):
        data_features = self.df[self.features]
        X = data_features.drop(self.target, axis=1)
        y = data_features[self.target]
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, y)
        return model

    def predict(self, input_data: dict):
        user_data = pd.DataFrame(input_data, index=[0])
        predicted_alcohol_consumption = self.model.predict(user_data)
        return predicted_alcohol_consumption[0]


if __name__ == "__main__":
    # 加載原始數據
    data = pd.read_csv('data/matrix_one_hot_encode.csv')

    # 初始化隨機森林模型
    features = ['paid[T.yes]', 'failures', 'absences', 'G1', 'G2', 'Dalc']
    rf_model = RandomForest(data, 'Dalc', features)

    # 互動性程式
    print("請輸入以下資料：")
    paid_yes = 0 # int(input("是否有支付課外輔導費用（1表示是，0表示否）："))
    failures = 4 # int(input("過去的失敗次數：（填1~3次，4表示否)"))
    absences = 0 #int(input("缺勤次數：(0~93)"))
    G1 = 20 #int(input("第一階段成績：(1~20)"))
    G2 = 20 #int(input("第二階段成績：(1~20)"))

    input_data = {
        'paid[T.yes]': paid_yes,
        'failures': failures,
        'absences': absences,
        'G1': G1,
        'G2': G2
    }

    # 進行預測
    predicted_dalc = rf_model.predict(input_data)
    print(f"預測的平日酒精消費量：{predicted_dalc}")

