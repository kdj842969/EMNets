import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

loss_full = pd.read_csv('./logs/run_.-tag-loss.csv')
val_full = pd.read_csv('./logs/run_.-tag-val_loss.csv')

loss = loss_full['Value']
val = val_full['Value']

xc = range(500)

plt.figure(1, figsize=(5, 10))
plt.plot(xc, loss)
plt.plot(xc, val)
plt.plot([100], [0.07506], marker='o', markersize=5, color="red")
plt.plot([200], [0.072313], marker='o', markersize=5, color="red")
plt.plot([500], [0.0685], marker='o', markersize=5, color="red")
plt.xlabel('num of epochs')
plt.ylabel('loss')
plt.title('train_loss vs. val_loss')
plt.grid(True)
plt.legend(['train', 'val', 'test'])
print(plt.style.available)
plt.style.use(['classic'])