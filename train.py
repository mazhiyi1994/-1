import torch.cuda

from torch import nn
from torch.optim import SGD
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms

from net import MyModel

writer = SummaryWriter(log_dir='logs')

transform = transforms.Compose([transforms.RandomHorizontalFlip(),
             transforms.RandomCrop(32, padding=4),
             transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
# トレーニングデータセット
train_data_set = datasets.CIFAR10("./dataset", train=True, transform=transform, download=True)

# テストデータセット
test_data_set = datasets.CIFAR10("./dataset", train=False, transform=transform, download=True)

train_data_size = len(train_data_set)
test_data_size = len(test_data_set)

print("訓練データ：{}, 検証データ:{}".format(train_data_size, test_data_size))
# データセットの読み込み
train_data_loader = DataLoader(train_data_set, batch_size=64,shuffle=True)
test_data_loader = DataLoader(test_data_set, batch_size=64)

# ネットワーク定義
myModel = MyModel()

# GPUを使用するかどうか
use_gpu = torch.cuda.is_available()
if (use_gpu):
    print("GPU使用可能")
    myModel = myModel.cuda()

# エポック数
epochs = 300
# 損失関数
lossFn = nn.CrossEntropyLoss()
# 最適化アルゴリズム
optimizer = SGD(myModel.parameters(), lr=0.01)
for epoch in range(epochs):
    print("エポック数 {}/{}".format(epoch + 1, epochs))

    # 損失変数
    train_total_loss = 0.0
    test_total_loss = 0.0
    # 精度
    train_total_acc = 0.0
    test_total_acc = 0.0

    # トレーニング開始
    for data in train_data_loader:
        inputs, labels = data
        if use_gpu:
            inputs = inputs.cuda()
            labels = labels.cuda()

        optimizer.zero_grad()  # 勾配を初期化
        outputs = myModel(inputs)  # 順伝播

        loss = lossFn(outputs, labels)  # 損失を計算
        # 精度を計算
        _, pred = torch.max(outputs, 1)  # 予測のインデックスを取得
        acc = torch.sum(pred == labels).item()  # 正解と比較して精度を取得
        loss.backward()  # 誤差逆伝播
        optimizer.step()  # パラメータの更新

        train_total_loss += loss.item()
        train_total_acc += acc

    # テスト
    with torch.no_grad():
        for data in test_data_loader:
            inputs, labels = data
            if use_gpu:
                inputs = inputs.cuda()
                labels = labels.cuda()

            outputs = myModel(inputs)  # 順伝播
            loss = lossFn(outputs, labels)  # 損失を計算
            # 精度を計算
            _, pred = torch.max(outputs, 1)  # 予測のインデックスを取得
            acc = torch.sum(pred == labels).item()  # 正解と比較して精度を取得

            test_total_loss += loss.item()
            test_total_acc += acc

    print("train loss:{},acc:{}. test loss:{},acc:{}".format(train_total_loss, train_total_acc / train_data_size,
                                                             test_total_loss, test_total_acc / test_data_size))
    writer.add_scalar('Loss/train', train_total_loss, epoch)
    writer.add_scalar('Loss/test', test_total_loss, epoch)
    writer.add_scalar('acc/train', train_total_acc / train_data_size, epoch)
    writer.add_scalar('acc/test', test_total_acc / test_data_size, epoch)
    if((epoch + 1)% 50 == 0 ):
        torch.save(myModel, "model/model_{}.pth".format(epoch+1))
# tensorboard --logdir=logs --port=6007
