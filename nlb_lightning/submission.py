import time

import torch

from nlb_tools.make_tensors import save_to_h5
from nlb_tools.evaluation import evaluate
import numpy as np

def text_save(filename, data):
    file = open(filename,'a')
    for i in range(len(data)):
        s = str(data[i]).replace('[','').replace(']','')
        s = s.replace("'",'').replace(',','') +'\n'
        file.write(s)
    file.close()

def make_submission(model, trainer, save_path):
    """Computes predictions for a given model and stores
    them in the EvalAI submission format for NLB.

    Parameters
    ----------
    model : pytorch_lightning.LightningModule
        A trained model that estimates firing rates.
    trainer : pytorch_lightning.Trainer
        The trainer corresponding to the trained model.
    save_path : str
        The destination file for estimated rates.

    """
    # Infer data shapes
    _, n_total, n_heldinout = trainer.datamodule.train_data[1].shape
    _, n_obs, n_heldin = trainer.datamodule.train_data[0].shape
    n_fwd = n_total - n_obs
    # Batch the data
    train_dataloader = trainer.datamodule.train_dataloader(shuffle=False)
    val_dataloader = trainer.datamodule.val_dataloader()
    # Restore the best model checkpoint
    best_model_path = trainer.checkpoint_callback.best_model_path
    model.load_from_checkpoint(best_model_path)
    def model_fwd(x):
         #data = torch.cat([x, x], dim=1)
         logrates, latents,_ = model.forward(x.to(model.device), use_logrates=True)
         return logrates, latents
    # Pass the batched data through the model
    train_rates, eval_rates = 0., 0.
    samples = 50
    for i in range(samples):
        train_rates += torch.cat([model_fwd(batch[0])[0].detach().cpu() for batch in train_dataloader])
        eval_rates += torch.cat([model_fwd(batch[0])[0].detach().cpu() for batch in val_dataloader])
    train_rates, eval_rates = train_rates/samples, eval_rates/samples
    train_rates, eval_rates = train_rates.exp().numpy(), eval_rates.exp().numpy()
    # Split model outputs for evaluation
    dataset_name = trainer.datamodule.hparams.dataset_name
    bin_width = trainer.datamodule.hparams.bin_width
    suffix = "" if (bin_width == 5) else f"_{int(bin_width)}"
    output_dict = {
        dataset_name
        + suffix: {
            "train_rates_heldin": train_rates[:, :-n_fwd, :n_heldin],
            "train_rates_heldout": train_rates[:, :-n_fwd, n_heldin:],
            "eval_rates_heldin": eval_rates[:, :-n_fwd, :n_heldin],
            "eval_rates_heldout": eval_rates[:, :-n_fwd, n_heldin:],
            "eval_rates_heldin_forward": eval_rates[:, -n_fwd:, :n_heldin],
            "eval_rates_heldout_forward": eval_rates[:, -n_fwd:, n_heldin:],
        }
    }
    # Save rates when the file becomes available (for parallel run compatibility)
    write_successful = False
    while not write_successful:
        try:
            save_to_h5(output_dict, save_path)
            write_successful = True
        except OSError:
            time.sleep(2)

def make_submission_lorentz(model, trainer, save_path):
    # Infer data shapes
    _, n_total, n_heldinout = trainer.datamodule.train_data[1].shape
    _, n_obs, n_heldin = trainer.datamodule.train_data[0].shape
    n_fwd = n_total - n_obs
    # Batch the data
    train_dataloader = trainer.datamodule.train_dataloader(shuffle=False)
    val_dataloader = trainer.datamodule.val_dataloader()
    # Restore the best model checkpoint
    best_model_path = trainer.checkpoint_callback.best_model_path
    model.load_from_checkpoint(best_model_path)

    def model_fwd(x):
        logrates, latents, _ = model.forward(x.to(model.device), use_logrates=True)
        return logrates, latents
    # Pass the batched data through the model
    train_rates, eval_rates = 0., 0.
    samples = 50
    for i in range(samples):
        train_rates += torch.cat([model_fwd(batch[0])[0].detach().cpu() for batch in train_dataloader])
        eval_rates += torch.cat([model_fwd(batch[0])[0].detach().cpu() for batch in val_dataloader])
    train_rates, eval_rates = train_rates / samples, eval_rates / samples
    train_rates, eval_rates = train_rates.exp().detach().cpu(), eval_rates.exp().detach().cpu()
    return None
