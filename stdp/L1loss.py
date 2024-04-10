def L1loss(model,l1_lamada,loss_fn,pred,y):
    l1_reg = torch.tensor(0., requires_grad=True)
    for param in model.parameters():
        l1_reg = l1_reg + torch.norm(param, 1)
    loss = loss_fn(pred,y)
    return loss+l1_lamada*l1_reg