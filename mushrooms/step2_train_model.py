import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from trains import Task, Logger


class MushRoomsDataset(Dataset):

    def __init__(self, csv_file):
        df = pd.read_csv(csv_file)

        self.categorical = [
            "cap_shape",
            "cap_surface",
            "cap_color",
            "bruises",
            "odor",
            "gill_attachment",
            "gill_spacing",
            "gill_size",
            "gill_color",
            "stalk_shape",
            "stalk_surface_above_ring",
            "stalk_surface_below_ring",
            "stalk_color_above_ring",
            "stalk_color_below_ring",
            "veil_type",
            "veil_color",
            "ring_number",
            "ring_type",
            "spore_print_color",
            "population",
            "habitat"
        ]
        self.target = "type"

        df.drop("stalk_root", axis=1, inplace=True)

        self.mushrooms_frame = pd.get_dummies(df, columns=self.categorical)

        self.X = self.mushrooms_frame.drop(self.target, axis=1)
        self.Y = self.mushrooms_frame[self.target].map({"p": 0, "e": 1})

    def __len__(self):
        return len(self.mushrooms_frame)

    def __getitem__(self, idx):
        if isinstance(idx, torch.Tensor):
            idx = idx.tolist()

        return [self.X.iloc[idx].values, self.Y[idx]]


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(112, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        x = x.squeeze()
        return x


def binary_acc(y_pred, y_test):
    y_pred_tag = torch.round(torch.sigmoid(y_pred))

    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum / y_test.shape[0]
    acc = torch.round(acc * 100)

    return acc


def train(model, device, train_loader, criterion, optimizer, epoch):
    epoch_loss = 0
    epoch_acc = 0
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data.float())
        loss = criterion(output, target.float())
        loss.backward()
        optimizer.step()

        acc = binary_acc(output, target.float())
        epoch_loss += loss.item()
        epoch_acc += acc.item()

        Logger.current_logger().report_scalar(
            "train", "loss", iteration=(epoch * len(train_loader) + batch_idx), value=loss.item())

    #print(f'Epoch(TRAIN) {epoch+0:03}: | Loss: {epoch_loss/len(train_loader):.8f} | Acc: {epoch_acc/len(train_loader):.3f}')


def test(model, device, criterion, test_loader, epoch):
    model.eval()
    epoch_loss = 0
    epoch_acc = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data.float())
            loss = criterion(output, target.float())

            acc = binary_acc(output, target.float())
            epoch_loss += loss.item()
            epoch_acc += acc.item()

        #print(f'Epoch(TEST) {epoch+0:03}: | Loss: {epoch_loss/len(test_loader):.8f} | Acc: {epoch_acc/len(test_loader):.3f}')

    Logger.current_logger().report_scalar("test", "loss", iteration=epoch, value=epoch_loss)
    Logger.current_logger().report_scalar("test", "accuracy", iteration=epoch, value=(acc / len(test_loader.dataset)))


task = Task.init(project_name="mushrooms", task_name="mushrooms step 2 train model")
args = {
    'stage_data_task_id': 'REPLACE_WITH_DATASET_TASK_ID',
}
task.connect(args)
task.execute_remotely()
dataset_task = Task.get_task(task_id=args["stage_data_task_id"])
dataset = MushRoomsDataset(dataset_task.artifacts["dataset"].get())
trainsize = int(0.8 * len(dataset))
testsize = len(dataset) - trainsize
trainset, testset = random_split(dataset, [trainsize, testsize])
trainloader = DataLoader(trainset, batch_size=128, shuffle=True)
testloader = DataLoader(testset, batch_size=32, shuffle=False)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = Net().to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

for epoch in range(1, 100):
    train(model, device, trainloader, criterion, optimizer, epoch)
    test(model, device, criterion, testloader, epoch)
