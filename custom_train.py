def custom_train:
  ct = 0
  for child in model2.children():
    ct += 1
    if ct < 23:
        for param in child.parameters():
            param.requires_grad = False

custom_train()
