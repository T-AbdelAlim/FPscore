import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot_loss(train_loss_values, val_loss_values, epoch, modelname):
    """
    Save a training/validation loss curve.

    Parameters
    ----------
    modelname : full path without extension, e.g. './results/my_exp/loss_plot'
                The function appends '.png' and saves there directly.
    """
    plt.figure(figsize=(10, 5))
    plt.title(f'Training and Validation Loss — Epoch {epoch + 1}')
    plt.plot(train_loss_values, label='train')
    plt.plot(val_loss_values,   label='val')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f'{modelname}.png')
    plt.close()
