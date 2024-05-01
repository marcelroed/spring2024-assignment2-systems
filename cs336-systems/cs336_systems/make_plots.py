import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('results/single_node_allreduce.csv', names=['Backend + Device', 'Tensor Size', 'World Size', 'Time'])
print(df)
for backend_device in df['Backend + Device'].unique():
    sub_df = df[df['Backend + Device'] == backend_device]
    plt.plot(sub_df['Tensor Size'], sub_df['Time'], label=backend_device)