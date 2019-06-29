in_feat=70
out_feat=10
h_feat=100
n_epoch=50
model3= torch.nn.Sequential(
    torch.nn.Linear(in_feat, h_feat),
    torch.nn.ReLU(),
    torch.nn.Linear(h_feat, out_feat),
    torch.nn.Softmax()
)

x_train=x_train.astype(float)
x_train=np.float32(x_train)

#define the loss function
criterion = torch.nn.BCEWithLogitsLoss()

#define thw optimizer used
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), 
                             lr=0.01)
for epoch in range(n_epoch):
    # Forward Propagation
    y_pred = model(torch.tensor(x_train))
    y_pred = y_pred.squeeze(1)  #for comparison with given targets

    # Compute and print loss
    loss = criterion(y_pred, torch.tensor(np.float32(y_train)))
    print('epoch: ', epoch,' loss: ', loss.item())
    # Zero the gradients
    optimizer.zero_grad()
    
    # perform a backward pass (backpropagation)
    loss.backward()
    
    # Update the parameters
    optimizer.step()
