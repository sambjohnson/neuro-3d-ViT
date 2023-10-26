

import os
import json
import shutil
import tempfile
import time
import datetime

import numpy as np
import nibabel as nib

from monai.losses import DiceLoss
from monai.inferers import sliding_window_inference
from monai import transforms
from monai.transforms import (
    AsDiscrete,
    Activations,
)

from monai.config import print_config
from monai.metrics import DiceMetric
from monai.utils.enums import MetricReduction
from monai.networks.nets import SwinUNETR
from monai import data
from monai.data import decollate_batch
from functools import partial
from monai.transforms import MapTransform

import torch
import einops


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = np.where(self.count > 0, self.sum / self.count, self.sum)


def datafold_read(datalist, basedir, fold=0, key="training"):
    with open(datalist) as f:
        json_data = json.load(f)

    json_data = json_data[key]

    for d in json_data:
        for k in d:
            if isinstance(d[k], list):
                d[k] = [os.path.join(basedir, iv) for iv in d[k]]
            elif isinstance(d[k], str):
                d[k] = os.path.join(basedir, d[k]) if len(d[k]) > 0 else d[k]

    tr = []
    val = []
    for d in json_data:
        if "fold" in d and d["fold"] == fold:
            val.append(d)
        else:
            tr.append(d)

    return tr, val


def save_checkpoint(model, epoch, filename="model.pt", best_acc=0, dir_add='', loss_list=None):
    state_dict = model.state_dict()
    save_dict = {"epoch": epoch, "best_acc": best_acc, "state_dict": state_dict, "loss": loss_list}
    filename = os.path.join(dir_add, filename)
    torch.save(save_dict, filename)
    print("Saving checkpoint", filename)

def get_loader(batch_size, data_dir, json_list, fold, roi, num_workers=8):
    data_dir = data_dir
    datalist_json = json_list
    
    train_files, validation_files = datafold_read(datalist=datalist_json, basedir=data_dir, fold=fold)
    train_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image", "label"], reader='NibabelReader'),
            AddChannelTransform(keys=["image"]),
            ConvertToMultiChannelBasedOnHCPClassesd(keys="label"),
            transforms.CropForegroundd(
                keys=["image", "label"],
                source_key="image",
                k_divisible=[roi[0], roi[1], roi[2]],
            ),
            transforms.RandSpatialCropd(
                keys=["image", "label"],
                roi_size=[roi[0], roi[1], roi[2]],
                random_size=False,
            ),
            transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
            transforms.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            #transforms.NormalizeIntensityd(keys="image", nonzero=True),
            transforms.RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
            transforms.RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image", "label"], reader='NibabelReader'),
            ConvertToMultiChannelBasedOnHCPClassesd(keys="label"),
            transforms.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        ]
    )

    train_ds = data.Dataset(data=train_files, transform=train_transform)

    train_loader = data.DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_ds = data.Dataset(data=validation_files, transform=val_transform)
    val_loader = data.DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader
    
class ConvertToMultiChannelBasedOnHCPClassesd(MapTransform):
    """
    Convert labels to multi channels based on brats classes:
    label 1 is the peritumoral edema
    label 2 is the GD-enhancing tumor
    label 3 is the necrotic and non-enhancing tumor core
    The possible classes are TC (Tumor core), WT (Whole tumor)
    and ET (Enhancing tumor).

    """

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            left_hemi = d[key][0].to(torch.int64)
            right_hemi = d[key][1].to(torch.int64) 
            lh_oh = torch.nn.functional.one_hot(left_hemi, num_classes=5).permute(3,0,1,2)
            rh_oh = torch.nn.functional.one_hot(right_hemi, num_classes=5).permute(3,0,1,2)
            w_background = torch.cat((lh_oh, rh_oh), axis=0).float()
            d[key] = w_background[[1,2,3,4,6,7,8,9]] ## Slices out background for both lh and rh [0, 5] indexes

        return d


class AddChannelTransform(MapTransform):
    """
    Add a size 1 dimension to accomodate 1-channel data.
    """

    def __init__(self, keys, channel_idx=0):
        super().__init__(keys=keys)
        self.channel_idx = channel_idx
        self.keys = keys

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            d[key] = torch.unsqueeze(d[key], self.channel_idx).float()
        return d
    
def train_epoch(model, loader, optimizer, epoch, loss_func, batch_size, max_epochs, device):
    model.train()
    start_time = time.time()
    run_loss = AverageMeter()
    for idx, batch_data in enumerate(loader):
        data, target = batch_data["image"].to(device), batch_data["label"].to(device)
        logits = model(data)
        loss = loss_func(logits, target)
        loss.backward()
        optimizer.step()
        run_loss.update(loss.item(), n=batch_size)
        print(
            "Epoch {}/{} {}/{}".format(epoch, max_epochs, idx, len(loader)),
            "loss: {:.4f}".format(run_loss.avg),
            "time {:.2f}s".format(time.time() - start_time),
        )
        start_time = time.time()
    return run_loss.avg


def val_epoch(
    model,
    loader,
    epoch,
    acc_func,
    max_epochs = None,
    device = torch.device("cuda"),
    model_inferer=None,
    post_sigmoid=None,
    post_pred=None,
):
    model.eval()
    start_time = time.time()
    run_acc = AverageMeter()

    with torch.no_grad():
        for idx, batch_data in enumerate(loader):
            data, target = batch_data["image"].to(device), batch_data["label"].to(device)
            logits = model_inferer(data)
            val_labels_list = decollate_batch(target)
            val_outputs_list = decollate_batch(logits)
            val_output_convert = [post_pred(post_sigmoid(val_pred_tensor)) for val_pred_tensor in val_outputs_list]
            acc_func.reset()
            acc_func(y_pred=val_output_convert, y=val_labels_list)
            acc, not_nans = acc_func.aggregate()
            run_acc.update(acc.cpu().numpy(), n=not_nans.cpu().numpy())
            dice_tc = run_acc.avg[0]
            dice_wt = run_acc.avg[1]
            dice_et = run_acc.avg[2]
            print(
                "Val {}/{} {}/{}".format(epoch, max_epochs, idx, len(loader)),
                ", dice_tc:",
                dice_tc,
                ", dice_wt:",
                dice_wt,
                ", dice_et:",
                dice_et,
                ", time {:.2f}s".format(time.time() - start_time),
            )
            start_time = time.time()

    return run_acc.avg

def trainer(model, 
            train_loader,
            val_loader, 
            optimizer, 
            loss_func, 
            acc_func, 
            scheduler, 
            batch_size,
            device = torch.device("cuda"),
            max_epochs = None, 
            val_every = 1,
            model_inferer=None, 
            start_epoch=0, 
            post_sigmoid=None, 
            post_pred=None  ):
      
    val_acc_max = 0.0
    dices_tc = []
    dices_wt = []
    dices_et = []
    dices_avg = []
    loss_epochs = []
    trains_epoch = []
    
    for epoch in range(start_epoch, max_epochs):
        print(time.ctime(), "Epoch:", epoch)
        epoch_time = time.time()
        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            max_epochs=max_epochs,
            epoch=epoch,
            loss_func=loss_func,
            batch_size=batch_size,
            device = torch.device("cuda"),

        )
        print(
            "Final training  {}/{}".format(epoch, max_epochs - 1),
            "loss: {:.4f}".format(train_loss),
            "time {:.2f}s".format(time.time() - epoch_time),
        )

        if (epoch + 1) % val_every == 0 or epoch == 0:
            loss_epochs.append(train_loss)
            trains_epoch.append(int(epoch))
            epoch_time = time.time()
            val_acc = val_epoch(
                model,
                val_loader,
                epoch=epoch,
                acc_func=acc_func,
                model_inferer=model_inferer,
                post_sigmoid=post_sigmoid,
                post_pred=post_pred,
            )
            val_avg_acc = np.mean(val_acc)
            print(
                "Final validation stats {}/{}".format(epoch, max_epochs - 1),
                "Dice_Avg:",
                val_avg_acc,
                ", time {:.2f}s".format(time.time() - epoch_time),
            )
            
            dices_avg.append(val_avg_acc)
            loss_list = [dices_avg]
            if val_avg_acc > val_acc_max:
                print("new best ({:.6f} --> {:.6f}). ".format(val_acc_max, val_avg_acc))
                val_acc_max = val_avg_acc
                save_checkpoint(
                    model,
                    epoch,
                    best_acc=val_acc_max,
                    loss_list = loss_list
                )
            scheduler.step()
    print("Training Finished !, Best Accuracy: ", val_acc_max)
    return (
        val_acc_max,
        dices_tc,
        dices_wt,
        dices_et,
        dices_avg,
        loss_epochs,
        trains_epoch,
    )
   
def main():
   # Data directories
    mount_point = "/home/b-parker/Desktop/neurocluster/home"
    json_list = f"{mount_point}/weiner/HCP/projects/CNL_scalpel/linux_aparc_fsav_VTC.json"


    roi = (128, 128, 128)
    # roi = (96, 96, 96) # changed
    batch_size = 2
    sw_batch_size = 4
    fold = 1
    infer_overlap = 0.5
    max_epochs = 100
    val_every = 10
    data_dir=''

    with open(json_list, 'r') as f:
        j = json.load(f)

    # inspect json file
    j['reader'] = 'NibabelReader'

    train_loader, val_loader = get_loader(batch_size,
                                        data_dir,
                                        json_list,
                                        fold,
                                        roi,
                                        num_workers=0)
    
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f'DEVICE == {device}')

    model = SwinUNETR(
        img_size=roi,
        in_channels=1,
        out_channels=8,
        feature_size=48,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        dropout_path_rate=0.0,
        use_checkpoint=True,
    ).to(device)

    torch.backends.cudnn.benchmark = True
    dice_loss = DiceLoss(to_onehot_y=False, sigmoid=True)
    post_sigmoid = Activations(sigmoid=True)
    post_pred = AsDiscrete(argmax=False, threshold=0.5)
    dice_acc = DiceMetric(include_background=True, reduction=MetricReduction.MEAN_BATCH, get_not_nans=True)
    model_inferer = partial(
        sliding_window_inference,
        roi_size=[roi[0], roi[1], roi[2]],
        sw_batch_size=sw_batch_size,
        predictor=model,
        overlap=infer_overlap,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5) ## TODO try 1e-3
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)


    load_pretrained_parmas = False

    if load_pretrained_parmas:
        data_dir = '/home/weiner/HCP/projects/CNL_scalpel'
        pretrained_model_path = f"{data_dir}/logs-5"
        best_checkpoint = torch.load(f'{pretrained_model_path}/best_model_neuro3d-ViT_01.pt')
        best_state_dict = best_checkpoint['state_dict']  # pretrained parameters
        state_dict = model.state_dict() # randomly intiaiized parameters of current model

        # This supervised model predicts a different output from the pretrained one,
        # so the outputs have a different semantic meaning and different shape.
        # Exclude the final weights from the the weight dictionary and then load

        load_state_dict = {k: v for k, v in best_state_dict.items() if 'conv3d_transpose_1' not in k}

        for k, v in state_dict.items():
            if k not in load_state_dict.keys():
                load_state_dict[k] = v

        model.load_state_dict(load_state_dict)

    start_epoch = 0

    (
    val_acc_max,
    dices_tc,
    dices_wt,
    dices_et,
    dices_avg,
    loss_epochs,
    trains_epoch,
    ) = trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        loss_func=dice_loss,
        acc_func=dice_acc,
        batch_size = batch_size,
        device = device,
        max_epochs=max_epochs,
        val_every = val_every,
        scheduler=scheduler,
        model_inferer=model_inferer,
        start_epoch=start_epoch,
        post_sigmoid=post_sigmoid,
        post_pred=post_pred,
    )


if __name__ == "__main__":
    main()
