class Merge(nn.Module):
    def __init__(self,modelA, modelB):
        super(Merge,self).__init__()
        self.modelA = modelA 
        self.modelB = modelB 
    
    def forward(self, x1):
        x2 = modelA(x1)
        x3 = modelB(x2)
        return x3
final_model = Merge(model2,model3)
    
x, y = Variable(torch.from_numpy(torch.tensor(x_train))), Variable(torch.from_numpy(y_train), requires_grad=False)

final_model(x.to(torch.long)
