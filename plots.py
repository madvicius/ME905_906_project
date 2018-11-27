import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
sn.set_palette("Paired")

plt.rcParams.update({'font.size': 24})

conv2 = pd.read_csv("plots/layer1/crossval_accu1.csv")[["train","val","time"]]
conv2["conv"] = '2 Camadas'

conv4 = pd.read_csv("plots/layer2/crossval_accu2.csv")[["train","val","time"]]
conv4["conv"] = '4 Camadas'


conv6 = pd.read_csv("plots/layer3/crossval_accu3.csv")[["train","val","time"]]
conv6["conv"] = '6 Camadas'

conv = conv2.append(conv4.append(conv6))

sn.set_style("whitegrid")
plt.figure(figsize=(16, 10))


fig, ax = plt.subplots(2,1)
conv_train = sn.catplot(y='train', x='conv', data=conv, kind="box", ax = ax[0])

sn.axes_style("whitegrid")
conv_val = sn.catplot(y='val', x='conv', data=conv, kind="box", ax = ax[1])

ax[0].set_xlabel('')
ax[0].set_ylabel('Treino')
ax[1].set_xlabel('')
ax[1].set_ylabel('Validação')

fig.suptitle("Acurácia e Camadas de Convolução",fontsize=24)
fig.savefig("plots/conv_acc.png")



bc = pd.read_csv("plots/batch_norm/crossval_accu_bnorm.csv")[["train","val","time"]]
bc["bnorm"] = "Sim"

nbc = pd.read_csv("plots/layer3/crossval_accu3.csv")[["train","val","time"]]
nbc['bnorm'] = 'Não'

bc_df = bc.append(nbc)

bcfig, ax = plt.subplots(2,1)


bctrain = sn.catplot(y='train', x='bnorm', data=bc_df, kind="box", ax = ax[0])
bcval = sn.catplot(y='val', x='bnorm', data=bc_df, kind="box", ax = ax[1])

ax[0].set_xlabel('')
ax[0].set_ylabel('Treino')
ax[1].set_xlabel('')
ax[1].set_ylabel('Validação')

bcfig.suptitle("Acurácia e Batch Normalization",fontsize=24)
bcfig.savefig("plots/bc_acc.png")



flatui = ["#0873EE", '#E11D1D']

sn.set_palette(sn.color_palette(flatui))

train_epoch = pd.read_csv("cnn_epoch.csv")[['acc','loss']]
train_epoch[''] = "Treino"
train_epoch['epoch'] = range(1,101)

val_epoch = pd.read_csv("cnn_epoch.csv")[['val_acc','val_loss']]
val_epoch.columns = ['acc','loss']
val_epoch[''] = "Validação"
val_epoch['epoch'] = range(1,101)


epoch = train_epoch.append(val_epoch)

fig,ax = plt.subplots(2)



plt.gcf().set_size_inches(11,8.5)

sn.lineplot(x='epoch',y = 'acc',hue='',data = epoch,ax = ax[0],legend=False)
sn.lineplot(x='epoch',y = 'loss',hue='',data = epoch,ax = ax[1])

ax[0].set_xlabel('')
ax[0].set_ylabel('Acurácia')
ax[1].set_xlabel('Épocas')
ax[1].set_ylabel('Perda')

fig.suptitle("Treinamento do Modelo",fontsize=24)
fig.savefig('plots/epoch.png',dpi=400)


