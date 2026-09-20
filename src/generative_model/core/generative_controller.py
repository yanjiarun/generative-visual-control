from src.generative_model.core.net import SinusoidalPosEmb
from torchvision.models import (
    squeezenet1_1, SqueezeNet1_1_Weights
)

import os
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.generative_model.core.datasets import (
    VectorFieldWithGradientsDataset, SparseVectorFieldWithGradientsDataset,
    RectangularMaskedDataset)



class FlowMatchingController(nn.Module):
    def __init__(self, image_size=128, action_dim=4, hidden_dim=128, history_length=1):
        super().__init__()
        self.action_dim = action_dim
        self.history_length = history_length
        
        # Use SqueezeNet as visual encoder
        self.visual_encoder = self._build_squeezenet_encoder(image_size)  # extracts 24-dimensional features from each image
        
         # Sequence feature extractor
        self.sequence_encoder = nn.Sequential(
            nn.Linear(24 * history_length, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 24)  # Output 24-dimensional feature vector
        )

        # Time embedding
        self.time_embed_dim = 16
        self.time_emb = nn.Sequential(
            SinusoidalPosEmb(self.time_embed_dim),
            nn.Linear(self.time_embed_dim, self.time_embed_dim * 4),
            nn.Mish(),
            nn.Linear(self.time_embed_dim * 4, self.time_embed_dim),
        )

        # Flow Matching network
        # Input dimension: (24-dimensional sequence features) + time embedding + current action
        input_dim = 24 + self.time_embed_dim + action_dim
        # input_dim = 64 + self.time_embed_dim + action_dim
        self.flow_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, action_dim),  # Predict 2-dimensional velocity field
            # nn.Tanh()  # Add Tanh activation to limit output to [-1, 1]   # remove Tanh, as it limits the range of the output
        )

    def _build_squeezenet_encoder(self, image_size):
        # Load pre-trained SqueezeNet model
        squeezenet = squeezenet1_1(weights=SqueezeNet1_1_Weights.IMAGENET1K_V1)
        
        # Remove the final classification layer, keep feature extraction part
        encoder = nn.Sequential(*list(squeezenet.features.children()))
        
        # Add adaptive pooling layer and flatten layer
        encoder.add_module("adaptive_pool", nn.AdaptiveAvgPool2d(1))
        encoder.add_module("flatten", nn.Flatten())
        
        # Calculate SqueezeNet output dimension
        with torch.no_grad():
            test_input = torch.randn(1, 3, image_size, image_size)
            test_output = encoder(test_input)
            original_dim = test_output.shape[1]
            print(f"SqueezeNet output dimension: {original_dim}")
        
        # Add downsampling layer (512 -> 24)
        encoder.add_module("downsample", nn.Sequential(
            nn.Linear(original_dim, 24),
            nn.ReLU()
        ))
        
        return encoder

    def forward(self, image_sequence, action_noise, t):
        """
        args:
            image_sequence: Input image sequence [B, history_length, C, H, W]
            action_noise: Noisy action [B, action_dim], range [-1, 1]
            t: Time step [B, 1]
        
        returns:
            predicted_flow: Predicted velocity field [B, action_dim], range [-1, 1]
        """
        batch_size = image_sequence.size(0)
        history_length = image_sequence.size(1)
        
        # [B, history_length, C, H, W] -> [B * history_length, C, H, W]
        combined_images = image_sequence.view(-1, *image_sequence.shape[2:])
        
        # output: [B * history_length, 24]
        combined_features = self.visual_encoder(combined_images)
        
        # reshape: [B * history_length, 24] -> [B, history_length, 24]
        sequence_features = combined_features.view(batch_size, history_length, -1)
        
        # reshape: [B, history_length, 24] -> [B, history_length * 24]
        sequence_features = sequence_features.view(batch_size, -1)

        sequence_encoding = self.sequence_encoder(sequence_features)
        
        # Time embedding: [B, 1] -> [B, 16]
        t_emb = self.time_emb(t)  # [B, 16]
        t_emb = t_emb.view(-1, self.time_embed_dim)  # [B, 16]

        # cat: [B, (24*history_length) + 16 + action_dim]
        condition = torch.cat([sequence_encoding, t_emb, action_noise], dim=1)  # [B, (24*history_length) + 16 + action_dim]

        # predict: [B, action_dim]
        predicted_flow = self.flow_net(condition)  # [B, action_dim]
        
        return predicted_flow

def flow_matching_loss(model, images, target_actions, device):

    batch_size = images.shape[0]

    # sample random time step t uniformly from [0, 1]
    t = torch.rand(batch_size, 1, device=device)
    
    # from prior distribution sample noise actions (uniform distribution [-1, 1])
    action_noise = torch.rand(batch_size, model.action_dim, device=device) * 2 - 1
    
    # linear interpolation: x_t = (1-t)*x_0 + t*x_1
    # here x_0 is noise actions, x_1 is target actions
    interpolated_actions = (1 - t) * action_noise + t * target_actions
    
    # true velocity field: u_t = x_1 - x_0
    true_flow = target_actions - action_noise
    
    # model predict velocity field
    predicted_flow = model(images, interpolated_actions, t)
        
    # calculate MSE loss
    loss = torch.mean((predicted_flow - true_flow) ** 2)
    
    return loss

def sample_action(model, images, num_steps=50, device="cpu", initial_actions=None):
    batch_size = images.shape[0]
    
    # from prior distribution sample noise actions (uniform distribution [-1, 1])
    actions = (torch.rand(batch_size, model.action_dim, device=device) * 2 - 1
               if initial_actions is None else initial_actions.to(device).clone())
    
    # use Euler method to solve ODE
    dt = 1.0 / num_steps
    for step in range(num_steps):
        t = torch.full((batch_size, 1), step / num_steps, device=device)
        flow = model(images, actions, t)
        actions = actions + flow * dt
        
        # ensure actions stay in [-1, 1] range
        # actions = torch.clamp(actions, -1.0, 1.0)
    
    return actions

def train_flow_matching(args):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Hyperparameters from command line arguments
    batch_size = args.batch_size
    epochs = args.epochs
    learning_rate = args.learning_rate
    vector_field_file = args.vector_field_file
    img_dir = args.img_dir
    save_dir = args.save_dir

    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(save_dir, f"logs_{timestamp}")
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)
    print(f"TensorBoard logs will be saved to: {log_dir}")

    # Create dataset and data loader
    if args.dataset_type == 'masked':
        print("Using masked dataset.")
        dataset = RectangularMaskedDataset(
        vector_field_file, 
        img_dir,
        mask_type=args.mask_type,
        )
    elif args.dataset_type == 'sparse':
        print("Using sparse dataset.")
        dataset = SparseVectorFieldWithGradientsDataset(
        vector_field_file, 
        img_dir
        )
    else:
        print("Using normal dataset.")
        dataset = VectorFieldWithGradientsDataset(
            vector_field_file, 
            img_dir
        )
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize model
    model = FlowMatchingController().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    best_loss = float('inf')

    # Training loop
    for epoch in range(epochs):
        total_loss = 0
        for images, target_actions in dataloader:
            images = images.to(device)
            target_actions = target_actions.to(device)

            # Compute loss
            loss = flow_matching_loss(model, images, target_actions, device)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{epochs}], Average Loss: {avg_loss:.4f}")
        writer.add_scalar('Loss/train_epoch', avg_loss, epoch+1)
        
        # Print training progress
        if (epoch + 1) % args.checkpoint_interval == 0:
            checkpoint_path = os.path.join(save_dir, f"checkpoint_epoch_{epoch+1}.pth")
            torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_model_path = os.path.join(save_dir, "flow_matching_best.pth")
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved with loss: {best_loss:.4f}")

    # Save model
    final_model_path = os.path.join(save_dir, "flow_matching_policy_final.pth")
    torch.save(model.state_dict(), final_model_path)
    print(f"Final model saved to {final_model_path}")

    writer.close()
    
    print("Training complete!")
    print(f"To view TensorBoard logs, run: tensorboard --logdir={log_dir}")

    save_loss_plot(log_dir, save_dir)
    
    return best_loss


def save_loss_plot(log_dir, save_dir):
    import matplotlib.pyplot as plt
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    
    plot_dir = os.path.join(save_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    
    event_acc = EventAccumulator(log_dir)
    event_acc.Reload()
    
    if 'Loss/train_epoch' in event_acc.Tags()['scalars']:
        loss_events = event_acc.Scalars('Loss/train_epoch')
        epochs_list = [int(e.step) for e in loss_events]
        loss_values = [e.value for e in loss_events]
        
        plt.figure(figsize=(10, 6))
        plt.plot(epochs_list, loss_values, 'b-', label='Training Loss')
        plt.title('Training Loss Curve')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        plot_path = os.path.join(plot_dir, 'training_loss_curve.png')
        plt.savefig(plot_path, bbox_inches='tight', dpi=300)
        plt.close()
        
        print(f"Loss curve saved to: {plot_path}")
    
    return plot_path
