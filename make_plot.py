import argparse
import matplotlib.pyplot as plt
import numpy as np
import sys
from tensorflow.python.summary.summary_iterator import summary_iterator


def load_tensorboard_data(event_file, keys):
    """ Load data from a TensorBoard log file. 
    Returns: List[Dict[key: value]], with keys from input keys plus 'step'
    """
    
    print(f"Reading from: {event_file}")
    
    data = []
    # Read the event file
    for event in summary_iterator(event_file):
        if event.summary:
            value_dict = {'step': event.step}
            for value in event.summary.value:
                if value.tag in keys:
                    value_dict[value.tag] = value.simple_value
            if set(keys).issubset(value_dict.keys()):
                # Only append if all keys are present
                data.append(value_dict)
    
    return data

def plot_losses(log_file, save_path=None, max_steps=None):
    """ Plot generator and discriminator losses from a TensorBoard log. """
    
    keys = ['G_-log_prob_', 'D_-log_prob_']
    data = load_tensorboard_data(log_file, keys)
    
    plt.figure(figsize=(24, 6))
    
    print(f"Loaded {len(data)} data points from {log_file}")
    print(f" item 0: {data[0] if data else 'No data'}")
    print(f" item 100: {data[100] if len(data) > 100 else 'No second data point'}")

    # Plot both losses
    # format: [[step, G_loss, D_loss], ...]
    if max_steps is None:
        max_steps = float('inf')
    step_g_d = np.array([[d['step'], d[keys[0]], d[keys[1]]] for d in data if d['step'] <= max_steps])

    plt.grid(True)
    # plt.plot(steps, g_losses, label='G Loss', alpha=0.7)
    # plt.plot(steps, d_losses, label='D Loss', alpha=0.7)
    plt.plot(step_g_d[:, 0], step_g_d[:, 1], label='G Loss', alpha=0.7)
    plt.plot(step_g_d[:, 0], step_g_d[:, 2], label='D Loss', alpha=0.7)

    # Set the limits around the data
    plt.xlim(min(step_g_d[:, 0]), max(step_g_d[:, 0]))
    
    plt.xlabel('Training Step')
    plt.ylabel('Loss')
    plt.title('GAN Training Losses')
    plt.legend()
    
    print(f"Total datapoints: {len(step_g_d)}")
    
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    if sys.stdout.isatty():
        plt.show()
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('log_file', type=str, default=None, 
                       help='Directory containing TensorBoard logs')
    parser.add_argument('--save-path', type=str, default='loss_plot.png',
                       help='Path to save the plot image')
    parser.add_argument('--max-steps', type=int, default=None,
                       help='Plot to this maximum step. Default: all steps')
    args = parser.parse_args()
    
    plot_losses(args.log_file, args.save_path, args.max_steps)

