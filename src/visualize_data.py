import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('data/HateTunisianData.csv')

plt.figure(figsize=(8, 6))
sns.countplot(x='index', data=data)
plt.title('Distribution of Hate Speech and Non-Hate Speech')
plt.xlabel('Label (0: Non-Hate Speech, 1: Hate Speech)')
plt.ylabel('Count')
plt.xticks(ticks=[0, 1], labels=['Non-Hate Speech', 'Hate Speech'])
plt.savefig('figures/class_distribution.png')
plt.show() 