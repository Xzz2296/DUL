
class density_loss(nn.Module):
    def __init__(self):
        super(density_loss, self).__init__()
        self.criterion = nn.CrossEntropyLoss()