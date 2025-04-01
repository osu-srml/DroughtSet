import os
import sys
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from util import AverageMeter, set_seed, DualWriter, create_windows
import DroughtSet.dataset as dataset
from model import build_model

def preprocess_agg(args,x_num_static_batch, x_num_time_batch, y_batch, mean_values, device):
    x_num_static_batch = x_num_static_batch.to(device)
    x_num_time_batch = x_num_time_batch.to(device)
    y_batch = y_batch.to(device)
    nan_mask_static = torch.isnan(x_num_static_batch)
    expanded_replacement_values = mean_values['static'].expand_as(x_num_static_batch)
    x_num_static_batch[nan_mask_static] = expanded_replacement_values[nan_mask_static]
    x_num_time_batch = x_num_time_batch.contiguous().view(x_num_time_batch.shape[0], x_num_time_batch.shape[1],x_num_time_batch.shape[2], -1)
    y_batch = y_batch.contiguous().view(y_batch.shape[0], y_batch.shape[1], y_batch.shape[2], -1)
    if(args.use_historic_y):
        x_num_time_batch = torch.cat((x_num_time_batch, y_batch), dim=2)
    current_x_num_static_batch = x_num_static_batch[:, 12, :].clone()
    current_x_num_time_batch = x_num_time_batch[:,12,:,:].clone().permute(2,0,1)

    y_batch = y_batch[:,12,:,:].permute(2,0,1)
    return current_x_num_static_batch,current_x_num_time_batch, y_batch, x_num_static_batch, x_num_time_batch
    
def run_epoch(loader, model, optimizer, loss_fn, mean_values, nan_mask_list, args, device, train=True):
    model.train() if train else model.eval()
    task_list = list(range(args.num_task))

    losses = [AverageMeter() for _ in range(args.num_task)]
    total_loss_meter = AverageMeter()

    p_bar = tqdm(enumerate(loader), total=len(loader)) if not args.no_progress else enumerate(loader)

    for batch_idx, (x_num_static_batch, x_cata_batch, x_num_time_batch, y_batch,agg_distance, original_index) in p_bar:
        current_x_num_static_batch, current_x_num_time_batch, y_batch, x_num_static_batch, x_num_time_batch = preprocess_agg(
            args, x_num_static_batch, x_num_time_batch, y_batch, mean_values, device
        )

        if (not train):
            f_distance = agg_distance.flatten(start_dim=1).to(device)
            f_distance[:, 12] = args.dis
            weighted_time_features, _ = model.agg(
                current_x_num_static_batch, x_num_static_batch, current_x_num_time_batch.permute(1,2,0), x_num_time_batch, distances=f_distance, show=True
            )
            current_x_num_time_batch = weighted_time_features.permute(2,0,1)

        x_num_time_window_split, y_batch_window_split, start_list, end_list = create_windows(
            current_x_num_time_batch, y_batch, args.input_window, args.unused_time, args.window_step
        )

        num_windows = max(int(len(x_num_time_window_split) * args.sample_ratio), 1) if train else len(x_num_time_window_split)
        selected_indices = np.random.choice(len(x_num_time_window_split), num_windows, replace=False) if train else range(num_windows)

        for window_idx in selected_indices:
            
            x_num_time_window = x_num_time_window_split[window_idx]
            
            if(train):
                f_distance = agg_distance.flatten(start_dim=1).to(device)
                f_distance[:,12] = args.dis
                weighted_time_features, att_weight = model.agg(current_x_num_static_batch, x_num_static_batch.to(device), current_x_num_time_batch.permute(1,2,0), x_num_time_batch.to(device) , distances = f_distance, show=True)
                
                x_num_time_window = weighted_time_features.permute(2,0,1)[start_list[window_idx]:start_list[window_idx]+args.input_window ]
            y_batch_window = y_batch_window_split[window_idx]
            with torch.set_grad_enabled(train):
                
                dec_inp = torch.cat([
                    x_num_time_window,
                    torch.zeros_like(current_x_num_time_batch[-args.unused_time:])
                ]).float().to(device)

                outputs = model(
                    x_num_time_window.to(device),
                    current_x_num_static_batch.to(device),
                    x_cata_batch.to(device),
                    tgt=dec_inp if args.use_decoder else None
                )

                all_task_loss = 0
                for task_id in task_list:
                    output_task = outputs[task_id].squeeze()
                    if output_task.dim() == 1:
                        output_task = output_task.unsqueeze(-1)

                    task_nan_mask = nan_mask_list[task_id][original_index, start_list[window_idx]+args.input_window:end_list[window_idx]].to(device)
                    target_task = y_batch_window[:, :, task_id].float().to(device).permute(1,0)
                    non_nan_mask = ~task_nan_mask

                    if non_nan_mask.sum() == 0:
                        continue

                    task_loss = loss_fn(output_task[non_nan_mask], target_task[non_nan_mask]).mean()
                    all_task_loss += task_loss
                    losses[task_id].update(task_loss.item())

                if all_task_loss == 0:
                    continue

                if train:
                    optimizer.zero_grad()
                    all_task_loss.backward()
                    optimizer.step()
                total_loss_meter.update(all_task_loss.item())

        if not args.no_progress:
            p_bar.set_description(f"{'Train' if train else 'Test'} Loss: {total_loss_meter.avg:.4f} | " +
                                  " | ".join([f"Loss_{task}: {loss.avg:.4f}" for task, loss in zip(['ESI','SIF','SM'], losses)]))


    if not args.no_progress and isinstance(p_bar, tqdm):
        p_bar.close()

    print(f"{'Train' if train else 'Test'} Total Loss: {total_loss_meter.avg:.4f}, " +
          ", ".join([f"{task} Loss: {loss.avg:.4f}" for task, loss in zip(['ESI','SIF','SM'], losses)]))

    
    
    
def main(args):
    
    

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset, test_dataset, train_loader, test_loader, mean_values, nan_mask_list = dataset.load_with_nan_mask(args.data_path, args.bs)
    
    for k,v in mean_values.items():
        mean_values[k] = v.to(device)      
    model = build_model(args, device)
        
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn_map = {
        'mse': nn.MSELoss(reduction='none'),
        'mae': nn.L1Loss(reduction='none')
    }
    loss_fn = loss_fn_map[args.loss.lower()]
        
    # Train
    for epoch in range(args.epochs):


        print(f"Epoch {epoch + 1}/{args.epochs}")

        train_loader = DataLoader(train_dataset, batch_size=args.bs, shuffle=True, num_workers=4, drop_last=True)
        run_epoch(train_loader, model, optimizer, loss_fn, mean_values, nan_mask_list, args, device, train=True)
        
        test_loader = DataLoader(test_dataset, batch_size=args.bs, shuffle=True, num_workers=4, drop_last=True)
        run_epoch(test_loader, model, optimizer, loss_fn, mean_values, nan_mask_list, args, device, train=False)

        if epoch % args.log_interval == 0:
            model_filename = os.path.join(args.save_path, f'model_epoch_{epoch}.pth')
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "args": args,
                },  model_filename )


        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    
    # ---------------------- Training settings ----------------------
    parser.add_argument('--epochs', type=int, default=10000)
    parser.add_argument('--bs', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--sample_ratio', type=float, default=1)
    parser.add_argument('--seed', type=int, default=11)
    parser.add_argument('--gpu', type=str, default="0")
    
    # ---------------------- log ----------------------
    parser.add_argument('--log_interval', type=int, default=1)
    parser.add_argument('--no_progress', action='store_true', default=False)

    # ---------------------- Model ----------------------
    parser.add_argument('--mlp_input', type=int, default=8)
    parser.add_argument('--mlp_hidden', type=int, default=10)
    parser.add_argument('--mlp_dim', type=int, default=16)
    parser.add_argument('--embedding_dim', type=int, default=4)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--num_d_layers', type=int, default=3)
    parser.add_argument('--dim_feedforward', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--inner_att_heads', type=int, default=4)
    parser.add_argument('--no_encoder_mask', action='store_true', default=False)
    parser.add_argument('--use_decoder', action='store_true', default=True)
    parser.add_argument('--loss', type=str, default="mae", choices=["mae", "mse", "MAE", "MSE"])
    parser.add_argument('--dis', type=float, default=0.8)

    # ---------------------- Data settings ----------------------
    parser.add_argument('--data_path', type=str, default="/users/PAS2353/tanxuwei99/code/project_climate/DroughtPrediction/release/SPDrought/temp")
    parser.add_argument('--save_path', type=str, default="debug/")
    parser.add_argument('--input_window', type=int, default=100)
    parser.add_argument('--unused_time', type=int, default=26)
    parser.add_argument('--window_step', type=int, default=26)
    parser.add_argument('--num_task', type=int, default=3)
    parser.add_argument('--time_dim', type=int, default=14)
    parser.add_argument('--time_rep_dim', type=int, default=48)
    parser.add_argument('--use_historic_y', action='store_true', default=True)



    
    args = parser.parse_args()


    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu 
    run_detail = f'debug_seed_{args.seed}'
    current_time = time.strftime("%Y%m%d-%H%M%S")
    args.save_path = os.path.join(args.save_path, run_detail, current_time)
    
    print(args.save_path, flush=True)
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    
    log_file_path = os.path.join(args.save_path, "log.txt")
    sys.stdout = DualWriter(log_file_path)
   
    print(args)
    set_seed(args)
    main(args)
    sys.stdout.log.close()
    sys.stdout = sys.stdout.terminal
