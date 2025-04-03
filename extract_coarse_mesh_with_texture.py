import argparse
from sugar_utils.general_utils import str2bool
from sugar_extractors.coarse_mesh import extract_mesh_and_texture_from_coarse_sugar

if __name__ == "__main__":
    # Parser
    parser = argparse.ArgumentParser(description='Script to extract a mesh and texture from a coarse SuGaR model.')
    parser.add_argument('-s', '--scene_path',
                        type=str, 
                        help='(Required) path to the scene data to use.')  # --OK
    parser.add_argument('-i', '--iteration_to_load', 
                        type=int, default=7000, 
                        help='iteration to load.')  # --OK
    parser.add_argument('-c', '--checkpoint_path', 
                        type=str, 
                        help='(Required) path to the vanilla 3D Gaussian Splatting Checkpoint to load.')  # --OK
    parser.add_argument('-m', '--coarse_model_path',
                        type=str, 
                        help='(Required) Path to the coarse model checkpoint.')  # --OK
    parser.add_argument('-o', '--mesh_output_dir',
                        type=str, 
                        default=None, 
                        help='path to the output directory.')  # --OK
    parser.add_argument('-n', '--n_gaussians_per_surface_triangle',
                        default=None, type=int, help='Number of gaussians per surface triangle.')  # --OK
    parser.add_argument('--square_size',
                        default=None, type=int, help='Size of the square to use for the texture.')  # --OK
    
    parser.add_argument('--eval', type=str2bool, default=True, help='Use eval split.')
    parser.add_argument('-g', '--gpu', type=int, default=0, help='Index of GPU to use.')
    
    args = parser.parse_args()
    
    # Call function
    extract_mesh_and_texture_from_coarse_sugar(args)
    
