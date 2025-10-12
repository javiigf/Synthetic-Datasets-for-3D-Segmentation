import subprocess
import argparse
import os
import re
import torch
import numpy as np
from models import create_model
from data import create_dataset
from options.test_options import TestOptions
import sys
from PIL import Image
import torch.nn.functional as F
import cv2
import torch_fidelity
import time

def find_max_epoch(checkpoints_path, model_name):
    """Finds the highest epoch number from checkpoint files in the given model's checkpoint directory."""
    model_checkpoint_dir = os.path.join(checkpoints_path, model_name)

    if not os.path.isdir(model_checkpoint_dir):
        raise ValueError(f"Checkpoint directory '{model_checkpoint_dir}' does not exist!")

    max_epoch = 0
    pattern = re.compile(r"(\d+)_net_D\.pth")  # Match filenames like 100_net_D.pth

    for filename in os.listdir(model_checkpoint_dir):
        match = pattern.match(filename)
        if match:
            epoch_num = int(match.group(1))
            max_epoch = max(max_epoch, epoch_num)

    if max_epoch == 0:
        raise ValueError(f"No valid checkpoint files found in '{model_checkpoint_dir}'!")

    return max_epoch

def count_images_in_trainA(dataroot):
    """Counts the number of images in trainA directory."""
    trainA_path = os.path.join(dataroot, "trainA")
    
    if not os.path.isdir(trainA_path):
        raise ValueError(f"'trainA' directory not found in '{dataroot}'!")

    image_count = sum(1 for f in os.listdir(trainA_path) if os.path.isfile(os.path.join(trainA_path, f)))

    if image_count == 0:
        raise ValueError(f"No images found in '{trainA_path}'!")

    return image_count


def run_test_epoch(model_name, dataroot, results_dir, checkpoints_path, device):
    
    start_time = time.time()
    
    """Runs the test for each epoch and logs the loss statistics."""
    max_epochs = find_max_epoch(checkpoints_path, model_name)
    num_test = count_images_in_trainA(dataroot)

    print(f"Detected {max_epochs} epochs from '{checkpoints_path}/{model_name}'.")
    print(f"Found {num_test} images in '{dataroot}/trainA'. Using device: {device}.")

    os.makedirs(os.path.join(results_dir, 'test_epochs'), exist_ok=True)  # Ensure directory exists
    os.makedirs(os.path.join(results_dir, "test_epochs", "real_A"), exist_ok=True)
    os.makedirs(os.path.join(results_dir, "test_epochs", "fake_B"), exist_ok=True)
    os.makedirs(os.path.join(results_dir, "test_epochs", "real_B"), exist_ok=True)

    loss_log_path = os.path.join(results_dir, f"{model_name}_losses.txt")
    
    with open(loss_log_path, "w") as loss_log:
        loss_log.write("epoch,mean_IoU,std_IoU,fid,ism,iss\n")
        
    for epoch_ix in range(1, max_epochs + 1):
        # print(f"Running test for epoch {epoch_ix}...")

        # Set up options for inference
        opt = TestOptions().parse()
        opt.name = model_name
        opt.dataroot = dataroot
        opt.results_dir = results_dir
        opt.epoch = str(epoch_ix)  # Ensure it's a string
        opt.num_test = num_test
        opt.model = "cut"
        opt.gpu_ids = [int(gpu.split(":")[-1]) for gpu in device.split(",")] if "cuda" in device else []
        opt.phase = "test"
        opt.serial_batches = True  # No shuffling for test mode
        opt.no_flip = True
        opt.eval = True
        
        # Load dataset
        dataset = create_dataset(opt)  
        if len(dataset) == 0:
            raise ValueError("Dataset is empty. Check dataroot and dataset format.")
        
        # Create and set up model
        model = create_model(opt)  
        model.setup(opt)  

        model.eval()  # Ensure model is in evaluation mode
        iou_scores = []  # Store IoU scores for all images
        nce_losses = []
        # Run inference
        for i, data in enumerate(dataset):
            model.set_input(data)  # Load data into model
            model.test()  # Perform inferenc

            visuals = model.get_current_visuals()
            image_path = model.get_image_paths()[0]  # Assuming batch size = 1
            filename = os.path.basename(image_path)
            
            real_A = visuals["real_A"].squeeze(0).cpu().numpy().transpose(1, 2, 0)
            fake_B = visuals["fake_B"].squeeze(0).cpu().numpy().transpose(1, 2, 0)
            real_A = (real_A + 1) / 2.0  # Normalize to [0, 1]
            real_A = (real_A * 255).astype(np.uint8)  # Convert to uint8 for saving as image
            fake_B = (fake_B + 1) / 2.0  # Normalize to [0, 1]
            fake_B = (fake_B * 255).astype(np.uint8)  # Convert to uint8 for saving as image

            # Apply Otsu's thresholding
            real_A_gray = cv2.cvtColor(real_A, cv2.COLOR_BGR2GRAY) if len(real_A.shape) == 3 else real_A
            fake_B_gray = cv2.cvtColor(fake_B, cv2.COLOR_BGR2GRAY) if len(fake_B.shape) == 3 else fake_B
            
            # Convert the images to 8-bit (CV_8UC1) to apply thresholding
            real_A_gray = np.uint8(real_A_gray * 255) if real_A_gray.dtype == np.float32 else real_A_gray
            fake_B_gray = np.uint8(fake_B_gray * 255) if fake_B_gray.dtype == np.float32 else fake_B_gray
            
            # Apply Otsu's thresholding
            _, real_mask = cv2.threshold(real_A_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            _, fake_mask = cv2.threshold(fake_B_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Convert masks to uint8 (0, 1) format
            real_mask = (real_mask > 0).astype(np.uint8)
            fake_mask = (fake_mask > 0).astype(np.uint8)
            # Compute IoU
            intersection = np.logical_and(real_mask, fake_mask).sum()
            union = np.logical_or(real_mask, fake_mask).sum()
            iou = intersection / union if union > 0 else 0
            iou_scores.append(iou)
            
            # Process real_A
            # real_A = visuals["real_A"].squeeze(0).cpu().numpy().transpose(1, 2, 0)
            # real_A = (real_A + 1) / 2.0  # Normalize to [0, 1]
            real_A_path = os.path.join(os.path.join(results_dir, "test_epochs", "real_A"), filename)
            real_A_path = real_A_path.replace('.jpg', '.png').replace('.jpeg', '.png')
            # real_A = (real_A * 255).astype(np.uint8)  # Convert to uint8 for saving as image
            Image.fromarray(real_A).save(real_A_path, format='PNG')  # Save as image file
            
            # Process fake_B
            # fake_B = visuals["fake_B"].squeeze(0).cpu().numpy().transpose(1, 2, 0)            
            # fake_B = (fake_B + 1) / 2.0  # Normalize to [0, 1]
            fake_B_path = os.path.join(os.path.join(results_dir, "test_epochs", "fake_B"), filename)
            fake_B_path = fake_B_path.replace('.jpg', '.png').replace('.jpeg', '.png')
            # fake_B = (fake_B * 255).astype(np.uint8)  # Convert to uint8 for saving as image
            Image.fromarray(fake_B).save(fake_B_path, format='PNG')  # Save as image file
            
            real_B = visuals["real_B"].squeeze(0).cpu().numpy().transpose(1, 2, 0)
            real_B = (real_B + 1) / 2.0  # Normalize to [0, 1]
            real_B_path = os.path.join(os.path.join(results_dir, "test_epochs", "real_B"), filename)
            real_B_path = real_B_path.replace('.jpg', '.png').replace('.jpeg', '.png')
            real_B = (real_B * 255).astype(np.uint8)  # Convert to uint8 for saving as image
            Image.fromarray(real_B).save(real_B_path, format='PNG')  # Save as image file
                   
            # print(f"Inference completed for: {image_path}")
            
        # Compute Mean and Std of IoU
        iou_mean = np.mean(iou_scores)
        iou_std = np.std(iou_scores)


        metrics = torch_fidelity.calculate_metrics(
            input1=os.path.join(results_dir, "test_epochs", "fake_B"),
            input2=os.path.join(results_dir, "test_epochs", "real_B"),
            isc=True,
            fid=True,
            verbose=True,
            device=device
        )
                        
        # Save results to log file
        with open(loss_log_path, "a") as loss_log:
            loss_log.write(f"{epoch_ix},{iou_mean:.4f},{iou_std:.4f},{metrics['frechet_inception_distance']:.4f}, {metrics['inception_score_mean']:.4f}, {metrics['inception_score_std']:.4f}\n")        

    # End time after the loop
    end_time = time.time()
    
    # Calculate time taken
    time_taken = end_time - start_time
    print(f"Time taken: {time_taken:.4f} seconds")
    with open(loss_log_path, "a") as loss_log:
        loss_log.write(f"Time taken: {time_taken:.4f} seconds\n")
    print(f"Loss statistics saved to {loss_log_path}")

def main():
    """Parses command-line arguments and runs the testing."""
    parser = argparse.ArgumentParser(description="Run test.py for each epoch in the model.")

    parser.add_argument("--name", required=True, help="Name of the model.")
    parser.add_argument("--dataroot", required=True, help="Directory containing test images.")
    parser.add_argument("--results_dir", required=True, help="Directory to save the results.")
    parser.add_argument("--checkpoints_dir", required=True, help="Base directory containing model checkpoints.")
    parser.add_argument("--gpu_ids", default="0", help="Comma-separated GPU IDs (e.g., '0,1') or 'cpu'.")

    args = parser.parse_args()

    # Determine device
    device = f"cuda:{args.gpu_ids}" if torch.cuda.is_available() and args.gpu_ids != "cpu" else "cpu"
    
    run_test_epoch(args.name, args.dataroot, args.results_dir, args.checkpoints_dir, device)


if __name__ == "__main__":
    # sys.argv = [
    #     '',
    #     '--name', 'cysts_20250310',  # Example model name
    #     '--dataroot', '/media/pedro/6TB/jesus/voronoiBasedGenerativeModel/CUT/datasets/cysts_test_epoch',  # Path to the dataset
    #     '--results_dir', '/media/pedro/6TB/jesus/voronoiBasedGenerativeModel/CUT/results',  # Path to save results
    #     '--checkpoints_dir', '/media/pedro/6TB/jesus/voronoiBasedGenerativeModel/CUT/checkpoints',  # Path to model checkpoints
    #     '--gpu_ids', '0'  # GPU ID or 'cpu'
    # ]
    main()
