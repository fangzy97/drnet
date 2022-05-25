import torch.optim as optim
import numpy as np
later_name = ['']
def get_1x_lr_params_NOscale(model):
    """
    This generator returns all the parameters of the net except for 
    the last classification layer. Note that for each batchnorm layer, 
    requires_grad is set to False in deeplab_resnet.py, therefore this function does not return 
    any batchnorm parameter
    """
    b = []

    b.append(model.netB.conv1)
    b.append(model.netB.bn1)
    b.append(model.netB.layer1)
    b.append(model.netB.layer2)
    b.append(model.netB.layer3)
    # b.append(model.netB.layer4)

    
    for i in range(len(b)):
        for j in b[i].modules():
            jj = 0
            for k in j.parameters():
                jj+=1
                if k.requires_grad:
                    yield k

def get_10x_lr_params(model):
    """
    This generator returns all the parameters for the last layer of the net,
    which does the classification of pixel into classes
    """
    b = []
    b.append(model.netB.layer5.parameters())
    b.append(model.netB.layer55.parameters())
    b.append(model.netB.layer6_0.parameters())
    b.append(model.netB.layer6_1.parameters())
    b.append(model.netB.layer6_2.parameters())
    b.append(model.netB.layer6_3.parameters())
    b.append(model.netB.layer6_4.parameters())
    b.append(model.netB.layer7.parameters())
    b.append(model.netB.residule1.parameters())
    b.append(model.netB.residule2.parameters())
    b.append(model.netB.residule3.parameters())
    b.append(model.netB.residule4.parameters())
    b.append(model.netB.layer9.parameters())

    for j in range(len(b)):
        for i in b[j]:
            yield i
            
            
def adjust_learning_rate(optimizer, i_iter):
    """Sets the learning rate to the initial LR divided by 5 at 60th, 120th and 160th epochs"""
    lr = lr_poly(0.00025, i_iter, 30000)
    optimizer.param_groups[0]['lr'] = lr
    optimizer.param_groups[1]['lr'] = lr * 10

def get_dconv_finetune_optimizer(args, model):
    lr = args.lr
    weight_list = []
    bias_list = []
    last_weight_list = []
    last_bias_list =[]
    first_weight_list = []
    first_bias_list = []
    second_weight_list = []
    second_bias_list = []
    center_list = []
    for name,value in model.named_parameters():
        #if 'cls' in name or 'p' in name or 'IOM' in name or 'combine' in name or 'center' in name:
        # if 'cls' in name or 'IOM' in name or 'dconv' in name or 'side' in name:
        if 'cls' in name or 'p' in name or 'IOM' in name or 'combine' in name or 'center' in name or 'non' in name or 'side' in name:
            print (name)
            if 'weight' in name:
                last_weight_list.append(value)
            elif 'bias' in name:
                last_bias_list.append(value)
        else:

            if 'mask' in name or 'name_offset' in name or 'name_weight' in name:
                # print (name)
                if 'weight' in name:
                    first_weight_list.append(value)
                elif 'bias' in name:
                    first_bias_list.append(value)
            else:
                if 'channel_downsample' in name:
                    if 'weight' in name:
                        second_weight_list.append(value)
                    elif 'bias' in name:
                        second_bias_list.append(value)
                else:
                    if 'weight' in name:
                        weight_list.append(value)
                    elif 'bias' in name:
                        bias_list.append(value)
    opt = optim.SGD([{'params': first_weight_list, 'lr':lr * 0.1},
                     {'params': first_bias_list, 'lr': lr * 0.2},
                     {'params': weight_list, 'lr':0},
                     {'params':bias_list, 'lr':0},
                     {'params': second_weight_list, 'lr': lr * 1},
                     {'params': second_bias_list, 'lr': lr * 2},
                     {'params':last_weight_list, 'lr':lr * 10},
                     {'params': last_bias_list, 'lr':lr * 20}], momentum=0.99, weight_decay=0.0005)

    return opt

def get_finetune_optimizer(args, model):
    lr = args.lr
    weight_list = []
    bias_list = []
    last_weight_list = []
    last_bias_list =[]
    first_weight_list = []
    first_bias_list = []
    center_list = []
    for name,value in model.named_parameters():
        #if 'cls' in name or 'p' in name or 'IOM' in name or 'combine' in name or 'center' in name:
        if 'cls' in name or 'IOM' in name or 'channel_downsample' in name:
        # if 'cls' in name or 'p' in name or 'IOM' in name or 'combine' in name or 'center' in name or 'non' in name or 'side' in name:
            print (name)
            if 'weight' in name:
                last_weight_list.append(value)
            elif 'bias' in name:
                last_bias_list.append(value)
        else:

            if 'mask' in name or 'name_offset' in name or 'name_weight' in name:
                print (name)
                if 'weight' in name:
                    first_weight_list.append(value)
                elif 'bias' in name:
                    first_bias_list.append(value)
            else:
                if 'weight' in name:
                    weight_list.append(value)
                elif 'bias' in name:
                    bias_list.append(value)
    opt = optim.SGD([{'params': first_weight_list, 'lr':lr * 0.1},
                     {'params': first_bias_list, 'lr': lr * 0.2},
                     {'params': weight_list, 'lr':lr},
                     {'params':bias_list, 'lr':lr*2},
                     {'params':last_weight_list, 'lr':lr * 10},
                     {'params': last_bias_list, 'lr':lr * 20}], momentum=0.99, weight_decay=0.0005)

    return opt

def get_finetune_optimizer2(args, model):
    lr = args.lr
    weight_list = []
    last_weight_list = []
    for name,value in model.named_parameters():
        if 'cls' in name:
            last_weight_list.append(value)
        else:
            weight_list.append(value)

    opt = optim.SGD([{'params': weight_list, 'lr':lr},
                     {'params':last_weight_list, 'lr':lr*10}], momentum=0.9, weight_decay=0.0005, nesterov=True)
    # opt = optim.SGD([{'params': weight_list, 'lr':lr},
    #                  {'params':last_weight_list, 'lr':lr*10}], momentum=0.9, nesterov=True)
    return opt

def get_optimizer(args, model):
    lr = args.lr
    opt = optim.SGD(params=model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0001)
    # lambda1 = lambda epoch: 0.1 if epoch in [85, 125, 165] else 1.0
    # scheduler = LambdaLR(opt, lr_lambda=lambda1)

    return opt

def lr_poly(base_lr, iter,max_iter,power=0.9):
    return base_lr*((1-float(iter)/max_iter)**(power))

def get_adam(args, model):
    lr = args.lr
    opt = optim.Adam(params=model.parameters(), lr =lr, weight_decay=0.0005)
    # opt = optim.Adam(params=model.parameters(), lr =lr)

    return opt

def reduce_lr(args, optimizer, epoch, factor=0.1):
    if 'voc' in args.dataset:
        change_points = [30,]
    else:
        change_points = None

    if change_points is not None and epoch in change_points:
        for g in optimizer.param_groups:
            g['lr'] = g['lr']*factor
            print(epoch, g['lr'])

def reduce_lr_poly(args, optimizer, global_iter, max_iter):
    base_lr = args.lr
    for g in optimizer.param_groups:
        g['lr'] = lr_poly(base_lr=base_lr, iter=global_iter, max_iter=max_iter, power=0.9)

def adjust_lr_2000(args, optimizer, global_step):
    if 'voc' in args.dataset:
        change_points = 2000
    else:
        change_points = None
    # else:

    # if epoch in change_points:
    #     lr = args.lr * 0.1**(change_points.index(epoch)+1)
    # else:
    #     lr = args.lr

    if global_step % 2000 == 0 and global_step > 0:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr']*0.1
            print (param_group['lr'])

def adjust_lr(args, optimizer, epoch):
    if 'cifar' in args.dataset:
        change_points = [80, 120, 160]
    elif 'indoor' in args.dataset:
        change_points = [60, 80, 100]
    elif 'dog' in args.dataset:
        change_points = [60, 80, 100]
    elif 'voc' in args.dataset:
        change_points = [30,]
    else:
        change_points = None
    # else:

    # if epoch in change_points:
    #     lr = args.lr * 0.1**(change_points.index(epoch)+1)
    # else:
    #     lr = args.lr

    if change_points is not None:
        change_points = np.array(change_points)
        pos = np.sum(epoch > change_points)
        lr = args.lr * (0.1**pos)
    else:
        lr = args.lr

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
