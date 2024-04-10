def count_weights_between(net, lower_bound, upper_bound):
    layers = [layer for layer in net.modules()]
    count_bull1 = (layers[1].weight.data.cpu().flatten()>lower_bound[0])&(layers[1].weight.data.cpu().flatten()<upper_bound[0])
    count_bull2 = (layers[2].weight.data.cpu().flatten() > lower_bound[1]) & (
                layers[2].weight.data.cpu().flatten() < upper_bound[1])
    count_bull3 = (layers[3].weight.data.cpu().flatten() > lower_bound[2]) & (
                layers[3].weight.data.cpu().flatten() < upper_bound[2])
    weight1_fiilter = ~((layers[1].weight.data.cpu()>lower_bound[0])&(layers[1].weight.data.cpu()<upper_bound[0]))
    weight2_fiilter = ~((layers[2].weight.data.cpu() > lower_bound[1]) & (layers[2].weight.data.cpu() < upper_bound[1]))
    weight3_fiilter = ~((layers[3].weight.data.cpu() > lower_bound[2]) & (layers[3].weight.data.cpu() < upper_bound[2]))
    weight1 = layers[1].weight.data.cpu()*weight1_fiilter
    weight2 = layers[2].weight.data.cpu() * weight2_fiilter
    weight3 = layers[3].weight.data.cpu() * weight3_fiilter
    count1 = count_bull1.float().sum()
    count2 = count_bull2.float().sum()
    count3 = count_bull3.float().sum()
    sum1,= layers[1].weight.data.cpu().flatten().size()
    sum2, = layers[2].weight.data.cpu().flatten().size()
    sum3, = layers[3].weight.data.cpu().flatten().size()
    count_all = count3+count2+count1
    sum_all = sum1+sum2+sum3
    print(f'layer1:{lower_bound[0]}-{upper_bound[0]} {count1}/{sum1}')
    print(f'layer2:{lower_bound[1]}-{upper_bound[1]} {count2}/{sum2}')
    print(f'layer3:{lower_bound[2]}-{upper_bound[2]} {count3}/{sum3}')
    print(f'sum:{sum_all},use_wight:{sum_all-count_all}')
    return count1,weight1,weight2,weight3
