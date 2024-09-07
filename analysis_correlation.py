import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('archive/T1.csv')

columns = ['LV ActivePower (kW)', 'Wind Speed (m/s)', 
                     'Theoretical_Power_Curve (KWh)', 'Wind Direction (°)']

correlation_matrix = df[columns].corr(method='pearson')

print(correlation_matrix)

plt.figure()
sns.heatmap(correlation_matrix, annot=True, vmin=-1, vmax=1)
plt.title('pearson matrix', fontsize=16)
plt.show()

# 분석 결과
"""
    Wind Speed (m/s) : 0.91
    Theoretical_Power_Curve (KWh) : 0.95
    Wind Direction (°) : -0.063
"""