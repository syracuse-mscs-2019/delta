import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

featFile = "./fbanks/p225_000.npy"
featFile2 = "./fbanks/p225_001.npy"

data = np.load(featFile)
data2 = np.load(featFile2)

print(np.array_equal(data, data2))

print('Shape', data.shape)

# Create the heatmap
dadj = data[:,:,0]
sns.heatmap(dadj)
plt.show()