import os.path
import torch
import argparse
from tqdm import tqdm
from dataloader import get_dataloader
from utils import Loss, train, test, save_best_record

from model.TPNet import network


def parse_args():
    parser = argparse.ArgumentParser(description='Let\'s go PVFormer')
    parser.add_argument('--save_path', default='/home/imt/Gaozzzz/pvd/TPNet/experiment')
    parser.add_argument('--max-epoch', type=int, default=2000)
    parser.add_argument('--lr', type=str, default=1e-4)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--num_workers', default=0, help='number of workers in dataloader')

    parser.add_argument('--modality', default='RGB', help='the type of the input, RGB,AUDIO, or MIX')
    parser.add_argument('--dataset', default='/home/user/imt/Gaozzzz/Data/colon_video_features', help='dataset to train on (default: )')
    return parser.parse_args()

if __name__ == '__main__':
    torch.cuda.set_device('cuda:0')
    exp_name = 'Exp_TPNet_1'
    num_blocks = 80

    best_AUC = 0
    best_AP = 0
    test_info = {"epoch": [], "Best_auc": [], "Best_ap": []}

    args = parse_args()
    save_path = os.path.join(args.save_path, exp_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    train_nloader, train_aloader, test_loader = get_dataloader(args)

    model = network(2048, flag="Train", nums_block=num_blocks)
    model = model.cuda()
    criterion = Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=0.00005)

    for step in tqdm(range(1, args.max_epoch + 1), total=args.max_epoch, dynamic_ncols=True):
        if (step - 1) % len(train_nloader) == 0:
            loadern_iter = iter(train_nloader)
        if (step - 1) % len(train_aloader) == 0:
            loadera_iter = iter(train_aloader)

        train(model, loadern_iter, loadera_iter, optimizer, criterion)

        if step % 5 == 0 and step > 10:
            auc, ap = test(test_loader, model)
            test_info["epoch"].append(step)
            test_info["Best_auc"].append(auc)
            test_info["Best_ap"].append(ap)

            if test_info["Best_auc"][-1] > best_AUC:
                best_AUC = test_info["Best_auc"][-1]
                torch.save(model.state_dict(), os.path.join(save_path, f'{exp_name}_Best_AUC.pkl'))
                print('auc : ' + str(test_info["Best_auc"][-1]))
                save_best_record(test_info, os.path.join(save_path, f'{exp_name}_Best_AUC_results.txt'))
            if test_info["Best_ap"][-1] > best_AP:
                best_AP = test_info["Best_ap"][-1]
                torch.save(model.state_dict(), os.path.join(save_path, f'{exp_name}_Best_AP.pkl'))
                print('ap : ' + str(test_info["Best_ap"][-1]))
                save_best_record(test_info, os.path.join(save_path, f'{exp_name}_Best_AP_results.txt'))