import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# 假设Y是分类变量，使用LabelEncoder将其转换为数值
label_encoder = LabelEncoder()
df['Education'] = label_encoder.fit_transform(df['Education'])

# 将分类变量进行独热编码
df_encoded = pd.get_dummies(df, columns=categorical_list)

# 分割特征和目标变量
X = df_encoded.drop('Education', axis=1)
Y = df_encoded['Education']

# 特征标准化
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 划分训练集和测试集
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# 创建线性回归模型
model = LinearRegression()

# 训练模型
model.fit(X_train, Y_train)

# 评估模型
Y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(Y_test, Y_pred))
print(f'Root Mean Squared Error on Test Set: {rmse}')

# 使用k-fold交叉验证评估模型性能
scores = cross_val_score(model, X, Y, cv=10, scoring='neg_mean_squared_error')
rmses = np.sqrt(-scores)
print("Root Mean Squared Error for each fold:", rmses)
print("Average RMSE:", np.mean(rmses))

# 可选: 查看模型系数
coefficients = pd.Series(model.coef_, index=X.columns)
coefficients_sorted = coefficients.abs().sort_values(ascending=False)
print("Top 10 Feature Coefficients:")
print(coefficients_sorted.head(10))

# 可选: 绘制预测值与真实值的散点图
plt.scatter(Y_test, Y_pred)
plt.xlabel("True Education Level")
plt.ylabel("Predicted Education Level")
plt.title("True vs Predicted Education Level")
plt.show()
