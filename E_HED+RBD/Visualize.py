import matplotlib.pyplot as plt
import pandas as pd

loss_path = r"./loss.txt"
mae_path = r"./mae.txt"
loss2_path = r"./loss_2.txt"
mae2_path = r"./mae_2.txt"

_loss = pd.read_table(loss_path).values
_mae = pd.read_table(mae_path).values

_loss2 = pd.read_table(loss2_path).values
_mae2 = pd.read_table(mae2_path).values

x1, y1 = _loss[:,0], _loss[:,1]
x2, y2 = _mae[:,0], _mae[:,1]
x3, y3 = _loss2[:,0], _loss2[:,1]
x4, y4 = _mae2[:,0], _mae2[:,1]
y3 = y3/8
plt.subplot(121)
plt.plot(x1,y1,color="red",linewidth=1, label="RBD_losscurve")
plt.plot(x3,y3,color="blue",linewidth=1, label="BRN_losscurve")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.legend()
plt.subplot(122)
plt.plot(x2,y2,color="red",linewidth=1, label="RBD_maecurve")
plt.plot(x4,y4,color="blue",linewidth=1, label="BRN_maecurve")
plt.xlabel("epoch")
plt.ylabel("mae")
plt.legend()
plt.show()