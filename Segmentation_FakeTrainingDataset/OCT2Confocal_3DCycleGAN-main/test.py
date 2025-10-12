"""General-purpose test script for image-to-image translation.

Once you have trained your model with train.py, you can use this script to test the model.
It will load a saved model from '--checkpoints_dir' and save the results to '--results_dir'.

It first creates model and dataset given the option. It will hard-code some parameters.
It then runs inference for '--num_test' images and save results to an HTML file.

Example (You need to train models first or download pre-trained models from our website):
    Test a CycleGAN model (both sides):
        python test.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan

    Test a CycleGAN model (one side only):
        python test.py --dataroot datasets/horse2zebra/testA --name horse2zebra_pretrained --model test --no_dropout

    The option '--model test' is used for generating CycleGAN results only for one side.
    This option will automatically set '--dataset_mode single', which only loads the images from one set.
    On the contrary, using '--model cycle_gan' requires loading and generating results in both directions,
    which is sometimes unnecessary. The results will be saved at ./results/.
    Use '--results_dir <directory_path_to_save_result>' to specify the results directory.

    Test a pix2pix model:
        python test.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/test_options.py for more test options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html
import tifffile as tiff
import numpy as np
import torch

try:
    import wandb
except ImportError:
    print('Warning: wandb package cannot be found. The option "--use_wandb" will result in error.')

def save_3d_tiff(img_data, img_path):
    """
    Save 3D numpy array as TIFF image.

    Parameters:
    - img_data: 3D numpy array containing image data
    - img_path: Path where the image will be saved
    """
    tiff.imsave(img_path, img_data)


if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 0
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers

    # initialize logger
    if opt.use_wandb:
        wandb_run = wandb.init(project=opt.wandb_project_name, name=opt.name, config=opt) if not wandb.run else wandb.run
        wandb_run._label(repo='CycleGAN-and-pix2pix')

    # create a website
    web_dir = os.path.join(opt.results_dir,os.path.basename(opt.dataroot),opt.name, '{}_{}'.format(opt.phase, opt.epoch))  # define the website directory
    if opt.load_iter > 0:  # load_iter is 0 by default
        web_dir = '{:s}_iter{:d}'.format(web_dir, opt.load_iter)
    print('creating web directory', web_dir)
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))
    # test with eval mode. This only affects layers like batchnorm and dropout.
    # For [pix2pix]: we use batchnorm and dropout in the original pix2pix. You can experiment it with and without eval() mode.
    # For [CycleGAN]: It should not affect CycleGAN as CycleGAN uses instancenorm without dropout.
    if opt.eval:
        model.eval()
    print(f"[TEST] Number of patches in dataset while testing: {len(dataset)}")
    # version 0 original
    # for i, data in enumerate(dataset):
    #     if i >= opt.num_test:  # only apply our model to opt.num_test images.
    #         break
    #     model.set_input(data)  # unpack data from data loader
    #     model.test()           # run inference
    #     visuals = model.get_current_visuals()  # get image results
    #     img_path = model.get_image_paths()     # get image paths

    #     # if i % 5 == 0:  # save images to an HTML file
    #     #     print('processing (%04d)-th image... %s' % (i, img_path))
    #     if i == len(dataset) - 1:  # check if it's the last image
    #         print('processing (%04d)-th image... %s' % (i, img_path))
    #     save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize, use_wandb=opt.use_wandb)

    # version 1: crops
    # filename_counts = {} 
    # import os
    # for i, data in enumerate(dataset):
    #     if i >= opt.num_test:
    #         break
        
    #     model.set_input(data)
    #     model.test()
    #     visuals = model.get_current_visuals()
        
    #     # img_path_originals is now correctly named to reflect it's a list
    #     img_path_originals = model.get_image_paths() 
    
    #     # Ensure img_path_originals is always a list, even if only one path is returned
    #     if not isinstance(img_path_originals, list):
    #         img_path_originals = [img_path_originals]
    
    #     # Iterate through each image path that the model returned
    #     final_img_paths_for_save = [] # Collect all unique paths for this 'data' item
    
    #     for original_path in img_path_originals:
    #         directory, filename = os.path.split(original_path)
    #         name, ext = os.path.splitext(filename)
    
    #         # Attempt to get the true base name (remove any existing _index if present)
    #         # This handles cases like 'image_1.tif' should become 'image' for counting
    #         base_name_parts = name.rsplit('_', 1)
    #         if len(base_name_parts) > 1 and base_name_parts[-1].isdigit():
    #             base_name = base_name_parts[0]
    #         else:
    #             base_name = name
    
    #         # Get the current count for this base name. If not seen, start with 1.
    #         # This gives us the *next* available index for this base_name
    #         count = filename_counts.get(base_name, 1)
    
    #         # Determine the final image path to use
    #         if count == 1:
    #             # First time seeing this base name, use the original path
    #             current_unique_path = original_path
    #         else:
    #             # Duplicate found, append the index
    #             current_unique_path = os.path.join(directory, f"{base_name}_{count}{ext}")
            
    #         # Add the unique path to our list for this batch
    #         final_img_paths_for_save.append(current_unique_path)
    
    #         # Increment the count for the next time we see this base name
    #         filename_counts[base_name] = count + 1
    
    #     # --- End of processing for individual paths within one 'data' item ---
    
    #     # For printing, just show the first path (or all if you prefer)
    #     if i == len(dataset) - 1:
    #         # Assuming you want to print just the first processed path for this data item
    #         print('processing (%04d)-th image... %s' % (i, final_img_paths_for_save[0] if final_img_paths_for_save else "No path found"))
        
    #     # Pass the potentially modified unique paths to save_images
    #     # IMPORTANT: Your `save_images` function must be able to handle `final_img_paths_for_save`
    #     # which is now always a list of paths. If save_images expects a single path, you might
    #     # need to loop here and call save_images for each item in final_img_paths_for_save
    #     # along with the corresponding visual.
        
    #     # Common scenario: save_images expects one path and one set of visuals per call
    #     # If `visuals` is also a list (e.g., [visual1, visual2]), you might do:
    #     # for j, path_to_save in enumerate(final_img_paths_for_save):
    #     #     # Assuming visuals[j] corresponds to path_to_save
    #     #     save_images(webpage, visuals[j], path_to_save, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize, use_wandb=opt.use_wandb)
        
    #     # If save_images can take a list of paths and visuals simultaneously, use this:
    #     save_images(webpage, visuals, final_img_paths_for_save, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize, use_wandb=opt.use_wandb)

    # version 1.2 crops cleaned
    filename_counts = {}
    import os
    realA_output_dir = os.path.join(web_dir, 'realA')
    fakeB_output_dir = os.path.join(web_dir, 'fakeB')
    os.makedirs(realA_output_dir, exist_ok=True)
    os.makedirs(fakeB_output_dir, exist_ok=True)

    for i, data in enumerate(dataset):
        if i >= opt.num_test:
            break
        
        model.set_input(data)
        model.test()
        visuals = model.get_current_visuals() # This dictionary typically contains {'real_A': ..., 'fake_B': ..., ...}
        
        # Extract only the 'fake_B' visual
        real_A_tensor = visuals['real_A']
        fake_B_tensor = visuals['fake_B']
        real_A_np = real_A_tensor.squeeze().cpu().numpy()
        fake_B_np = fake_B_tensor.squeeze().cpu().numpy()
        
        # img_path_originals is now correctly named to reflect it's a list
        img_path_originals = model.get_image_paths() 
        
        # Ensure img_path_originals is always a list, even if only one path is returned
        if not isinstance(img_path_originals, list):
            img_path_originals = [img_path_originals]
        
        # Iterate through each image path that the model returned
        final_img_paths_for_save_fake_B = [] # Collect all unique paths for this 'data' item
        final_img_paths_for_save_real_A = [] # Collect all unique paths for this 'data' item

        for original_path in img_path_originals:
            directory, filename = os.path.split(original_path)
            name, ext = os.path.splitext(filename)
            
            # Attempt to get the true base name (remove any existing _index if present)
            # This handles cases like 'image_1.tif' should become 'image' for counting
            base_name_parts = name.rsplit('_', 1)
            if len(base_name_parts) > 1 and base_name_parts[-1].isdigit():
                base_name = base_name_parts[0]
            else:
                base_name = name
            
            # Get the current count for this base name. If not seen, start with 1.
            # This gives us the *next* available index for this base_name
            count = filename_counts.get(base_name, 1)
            
            # Determine the final image path to use
            # The save_images function will typically append "_fake_B" (or similar suffix) itself
            # based on the keys in the 'visuals' dictionary you pass to it.
            if count == 1:
                # current_unique_path_fake_B = os.path.join(directory, 'fake_B')
                # current_unique_path_real_A = os.path.join(directory, 'real_A')
                realA_save_path = os.path.join(realA_output_dir, f'{base_name}.tif')
                fakeB_save_path = os.path.join(fakeB_output_dir, f'{base_name}.tif')
            else:
                # current_unique_path_fake_B = os.path.join(directory, 'fake_B', f"{base_name}_{count}{ext}")
                # current_unique_path_real_A = os.path.join(directory, 'real_A', f"{base_name}_{count}{ext}")
                realA_save_path = os.path.join(realA_output_dir, f'{base_name}_{count}.tif')
                fakeB_save_path = os.path.join(fakeB_output_dir, f'{base_name}_{count}.tif')


            # Increment the count for the next time we see this base name
            filename_counts[base_name] = count + 1
            
        
        # --- End of processing for individual paths within one 'data' item ---
        
        # For printing, just show the first path (or all if you prefer)
        if i == len(dataset) - 1:
            print('processing (%04d)-th image... %s' % (i, final_img_paths_for_save_fake_B[0] if final_img_paths_for_save_fake_B else "No path found"))
            print('processing (%04d)-th image... %s' % (i, final_img_paths_for_save_real_A[0] if final_img_paths_for_save_real_A else "No path found"))

        tiff.imsave(realA_save_path, real_A_np)
        tiff.imsave(fakeB_save_path, fake_B_np)
        # Pass ONLY the 'fake_B' visual to save_images
        # save_images(webpage, fake_B_to_save, final_img_paths_for_save_fake_B, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize, use_wandb=opt.use_wandb)
        # save_images(webpage, real_A_to_save, final_img_paths_for_save_real_A, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize, use_wandb=opt.use_wandb)

    # # ## version 2 reconstruct
    # reconstruction_data_collector = {}
    # reconstruction_data_collector = {}

    # def reconstruct_volume(patches, coords, reconstruction_shape_DHW, final_output_dtype):
    #     """
    #     Reconstructs a single 3D volume from a list of patches and their coordinates.
    #     Overlapping regions are handled by overwriting (the last patch placed takes precedence).

    #     Args:
    #         patches (list): List of NumPy arrays, each representing a generated patch.
    #                         Expected shape: (D_patch, H_patch, W_patch).
    #         coords (list): List of tuples, each (z_start, h_start, w_start),
    #                         representing the start indices in (Depth, Height, Width) order.
    #         reconstruction_shape_DHW (tuple): The (Depth, Height, Width) dimensions of the canvas for reconstruction.
    #         final_output_dtype (np.dtype): The desired data type for the reconstructed volume (e.g., np.uint16).

    #     Returns:
    #         np.ndarray: The reconstructed 3D volume.
    #     """
    #     if not patches:
    #         return None

    #     full_D_trans, full_H_trans, full_W_trans = reconstruction_shape_DHW

    #     # Initialize the full volume with zeros. We use np.float32 for intermediate calculations
    #     # as model outputs are typically float, before final scaling and casting.
    #     reconstructed_volume = np.zeros((full_D_trans, full_H_trans, full_W_trans), dtype=np.float32)

    #     # Iterate through patches and their coordinates
    #     for patch, (z_start, h_start, w_start) in zip(patches, coords):
    #         # Ensure the patch is a NumPy array, squeezing singleton dimensions
    #         if isinstance(patch, torch.Tensor):
    #             patch = patch.squeeze().cpu().numpy()
            
    #         # Get dimensions of the current patch (Depth, Height, Width)
    #         patch_D, patch_H, patch_W = patch.shape

    #         # Calculate the end coordinates for placing the patch
    #         z_end = z_start + patch_D
    #         h_end = h_start + patch_H
    #         w_end = w_start + patch_W

    #         # Place the patch into the reconstructed volume by DIRECTLY ASSIGNING.
    #         # This means the last patch written to an overlapping region will determine the final pixel value.
    #         reconstructed_volume[z_start:z_end, h_start:h_end, w_start:w_end] = patch

    #     # Scale model output from assumed [-1, 1] range to [0, 1]
    #     reconstructed_volume = (reconstructed_volume + 1) / 2.0 
        
    #     # Scale from [0, 1] to the full range of the target dtype (e.g., 0-65535 for uint16)
    #     # Determine min/max values based on the final_output_dtype
    #     info = np.iinfo(final_output_dtype) if np.issubdtype(final_output_dtype, np.integer) else np.finfo(final_output_dtype)
    #     min_val, max_val = info.min, info.max
        
    #     reconstructed_volume = reconstructed_volume * (max_val - min_val) + min_val
        
    #     # Clip values to ensure they are within the valid range of the target dtype
    #     reconstructed_volume = np.clip(reconstructed_volume, min_val, max_val)
        
    #     # Cast the reconstructed volume to the final desired data type
    #     return reconstructed_volume.astype(final_output_dtype)

    # # --- Main testing loop ---
    # # Initialize output directory for reconstructed volumes
    # output_dir = os.path.join(opt.results_dir, opt.phase + '_reconstructed')
    # os.makedirs(output_dir, exist_ok=True)
    
    # reconstructed_volume_counts = {} # To keep track of multiple reconstructed volumes from the same base name
    
    
    # # --- Main testing loop ---
    # # Initialize before the loop, as `opt` should be available here
    # output_dir = os.path.join(opt.results_dir, opt.phase + '_reconstructed')
    # os.makedirs(output_dir, exist_ok=True)
    
    # reconstructed_volume_counts = {}
    
    
    # for i, data in enumerate(dataset):
    #     if i >= opt.num_test:
    #         break
        
    #     model.set_input(data)
    #     model.test()
    #     visuals = model.get_current_visuals()
        
    #     original_volume_path = data['A_paths'][0]
        
    #     # Extract raw coordinates from dataset (x_start_H, y_start_W, z_start_D, W_trans, H_trans, D_trans)
    #     coords_tensor = data['A_coords'] 
    #     patch_coords_raw = coords_tensor.cpu().numpy().tolist()[0] 
        
    #     # Unpack based on dataset's ordering
    #     x_start_H_dataset, y_start_W_dataset, z_start_D_dataset, total_W_trans, total_H_trans, total_D_trans = patch_coords_raw
    
    #     # Reorder coordinates to (Depth, Height, Width) for NumPy array indexing
    #     patch_start_coords_DHW = (z_start_D_dataset, x_start_H_dataset, y_start_W_dataset)
    
    #     # The shape of the canvas for reconstruction (Depth, Height, Width)
    #     reconstruction_canvas_shape_DHW = (total_D_trans, total_H_trans, total_W_trans)
    
    #     original_volume_dims_tensor = data['A_original_shape']
    #     original_volume_dims = original_volume_dims_tensor.cpu().numpy().tolist()[0]
        
    #     original_data_type_str = data['A_dtype'][0]
    #     final_output_dtype = np.dtype(original_data_type_str)
    
    #     generated_patch_tensor = visuals['fake_B'] 
    #     generated_patch_np = generated_patch_tensor.squeeze().cpu().numpy()
        
    #     # Initialize entry for this volume if it's the first time we see it
    #     if original_volume_path not in reconstruction_data_collector:
    #         reconstruction_data_collector[original_volume_path] = {
    #             'patches': [],
    #             'coords': [], # This list will store (z_start, h_start, w_start)
    #             'reconstruction_shape_DHW': reconstruction_canvas_shape_DHW,
    #             'final_output_dtype': final_output_dtype,
    #             'original_volume_dims': original_volume_dims
    #         }
        
    #     # Store the patch and its correctly ordered (D, H, W) start coordinates
    #     reconstruction_data_collector[original_volume_path]['patches'].append(generated_patch_np)
    #     reconstruction_data_collector[original_volume_path]['coords'].append(patch_start_coords_DHW)
    
    #     if i == len(dataset) - 1:
    #         print('Finished processing all patches. Starting 3D volume reconstruction...')
    
    # # --- After the main loop, reconstruct and save full volumes ---
    # for original_vol_path, data_for_reconstruction in reconstruction_data_collector.items():
    #     print(f"Reconstructing volume: {original_vol_path}")
    #     reconstructed_3d_volume = reconstruct_volume(
    #         data_for_reconstruction['patches'],
    #         data_for_reconstruction['coords'],
    #         data_for_reconstruction['reconstruction_shape_DHW'],
    #         data_for_reconstruction['final_output_dtype']
    #     )
    
    #     if reconstructed_3d_volume is not None:
    #         directory, filename = os.path.split(original_vol_path)
    #         name, ext = os.path.splitext(filename)
    
    #         base_name_parts = name.rsplit('_', 1)
    #         if len(base_name_parts) > 1 and base_name_parts[-1].isdigit():
    #             base_name = base_name_parts[0]
    #         else:
    #             base_name = name
    
    #         count = reconstructed_volume_counts.get(base_name, 1)
    
    #         orig_W, orig_H, orig_D = data_for_reconstruction['original_volume_dims']
    #         if count == 1:
    #             final_save_filename = f"{base_name}_reconstructed_{orig_D}x{orig_H}x{orig_W}{ext}"
    #         else:
    #             final_save_filename = f"{base_name}_reconstructed_{count}_{orig_D}x{orig_H}x{orig_W}{ext}"
            
    #         reconstructed_volume_counts[base_name] = count + 1
    
    #         save_path = os.path.join(output_dir, final_save_filename)
    #         print(f"DEBUG: Original Volume Path: {original_volume_path}")
    #         print(f"DEBUG: Patch raw coords from dataset: {patch_coords_raw}")
    #         print(f"DEBUG: Reordered patch start coords (D, H, W): {patch_start_coords_DHW}")
    #         print(f"DEBUG: Reconstruction canvas shape (D, H, W): {reconstruction_canvas_shape_DHW}")
    #         print(f"DEBUG: Generated patch NumPy shape: {generated_patch_np.shape}")
    #         print(f"Saving reconstructed volume to: {save_path}")
    #         tiff.imwrite(save_path, reconstructed_3d_volume)
    #         print(f"DEBUG: Reconstructed volume dtype: {reconstructed_3d_volume.dtype}")
    #         print(f"DEBUG: Reconstructed volume min/max: {reconstructed_3d_volume.min()}/{reconstructed_3d_volume.max()}")
    #     else:
    #         print(f"Could not reconstruct volume from {original_vol_path}: No patches found.")
    
    # print("All 3D volumes reconstructed and saved.")
    
    webpage.save()  # save the HTML