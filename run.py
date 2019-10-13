
import torch
from world import World
from tqdm import tqdm
import torch.optim as optim
from tensorboardX import SummaryWriter

writer = SummaryWriter()
device = 'cpu'
# if torch.cuda.is_available():
#     device = 'cuda'
print(device)
x1 = torch.rand(16,16,16).to(device)
x2 = torch.rand(16,16,16).to(device)
x3 = torch.rand(16,16,16).to(device)
model = World(x1,x2,x3,device=device).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
model.train()
model.init()
for epoch in tqdm(range(1, 500 + 1)):
    loss = model.forward()
    loss.backward()
    optimizer.step()

    for name, param in model.named_parameters():
        writer.add_histogram(name, param.clone().cpu().data.numpy(), epoch)


    for i, sp in enumerate(model.spices):
        writer.add_histogram(str(i)+"_score", torch.Tensor(sp.score).clone().cpu().data.numpy(), epoch)
