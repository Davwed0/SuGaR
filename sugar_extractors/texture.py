import numpy as np
import torch
from sugar_scene.sugar_model import SuGaR
from sugar_utils.spherical_harmonics import SH2RGB
from pytorch3d.structures import Meshes
from pytorch3d.renderer import TexturesUV
from sugar_utils.mesh_rasterization import MeshRasterizer, RasterizationSettings
import os
import imageio


@torch.no_grad()
def compute_textured_mesh_for_sugar_mesh(
    sugar: SuGaR,
    square_size: int = 10,
    n_sh=0,
    texture_with_gaussian_renders=True,
    bg_color=[0.0, 0.0, 0.0],
):
    device = sugar.device

    if sugar.nerfmodel is None:
        raise ValueError("You must provide a NerfModel to use this function.")

    if square_size < 3:
        raise ValueError("square_size must be >= 3")

    surface_mesh = sugar.surface_mesh
    faces = surface_mesh.faces_packed()

    n_triangles = len(faces)
    n_squares = n_triangles // 2 + 1
    n_square_per_axis = int(np.sqrt(n_squares) + 1)
    texture_size = square_size * (n_square_per_axis)

    # Build faces UV.
    # Each face will have 3 corresponding vertices in the UV map
    faces_uv = torch.arange(3 * n_triangles, device=device).view(
        n_triangles, 3
    )  # n_triangles, 3

    # Build corresponding vertices UV
    vertices_uv = torch.cartesian_prod(
        torch.arange(n_square_per_axis, device=device),
        torch.arange(n_square_per_axis, device=device),
    )
    bottom_verts_uv = torch.cat(
        [
            vertices_uv[n_square_per_axis:-1, None],
            vertices_uv[: -n_square_per_axis - 1, None],
            vertices_uv[n_square_per_axis + 1 :, None],
        ],
        dim=1,
    )
    top_verts_uv = torch.cat(
        [
            vertices_uv[1:-n_square_per_axis, None],
            vertices_uv[: -n_square_per_axis - 1, None],
            vertices_uv[n_square_per_axis + 1 :, None],
        ],
        dim=1,
    )

    vertices_uv = torch.cartesian_prod(
        torch.arange(n_square_per_axis, device=device),
        torch.arange(n_square_per_axis, device=device),
    )[:, None]
    u_shift = torch.tensor([[1, 0]], dtype=torch.int32, device=device)[:, None]
    v_shift = torch.tensor([[0, 1]], dtype=torch.int32, device=device)[:, None]
    bottom_verts_uv = torch.cat(
        [vertices_uv + u_shift, vertices_uv, vertices_uv + u_shift + v_shift], dim=1
    )
    top_verts_uv = torch.cat(
        [vertices_uv + v_shift, vertices_uv, vertices_uv + u_shift + v_shift], dim=1
    )

    verts_uv = torch.cat([bottom_verts_uv, top_verts_uv], dim=1)
    verts_uv = verts_uv * square_size
    verts_uv[:, 0] = verts_uv[:, 0] + torch.tensor([[-2, 1]], device=device)
    verts_uv[:, 1] = verts_uv[:, 1] + torch.tensor([[2, 1]], device=device)
    verts_uv[:, 2] = verts_uv[:, 2] + torch.tensor([[-2, -3]], device=device)
    verts_uv[:, 3] = verts_uv[:, 3] + torch.tensor([[1, -1]], device=device)
    verts_uv[:, 4] = verts_uv[:, 4] + torch.tensor([[1, 3]], device=device)
    verts_uv[:, 5] = verts_uv[:, 5] + torch.tensor([[-3, -1]], device=device)

    verts_uv = verts_uv.reshape(-1, 2) / texture_size
    print("Building UV map done.")

    # Get, for each pixel, the corresponding face
    uvs_coords = torch.cartesian_prod(
        torch.arange(texture_size, device=device, dtype=torch.int32),
        torch.arange(texture_size, device=device, dtype=torch.int32),
    ).view(texture_size, texture_size, 2)

    square_of_uvs = uvs_coords // square_size
    square_of_uvs = square_of_uvs[..., 0] * n_square_per_axis + square_of_uvs[..., 1]

    uvs_in_top_triangle = uvs_coords % square_size
    uvs_in_top_triangle = uvs_in_top_triangle[..., 0] < uvs_in_top_triangle[..., 1]

    uv_to_faces = 2 * square_of_uvs + uvs_in_top_triangle
    uv_to_faces = uv_to_faces.transpose(0, 1).clamp_max(n_triangles - 1)

    # Build Texture
    texture_img = torch.zeros(texture_size, texture_size, 3, device=device)
    texture_count = torch.zeros(texture_size, texture_size, 1, device=device)

    # Create textures for base_color and roughness as well
    texture_base_color = torch.zeros(texture_size, texture_size, 3, device=device)
    texture_roughness = torch.zeros(texture_size, texture_size, 1, device=device)

    # Average color of visited faces
    face_colors = torch.zeros(n_triangles, 3, device=device)
    face_count = torch.zeros(n_triangles, 1, device=device)
    face_base_colors = torch.zeros(n_triangles, 3, device=device)
    face_roughness = torch.zeros(n_triangles, 1, device=device)

    # Color of non visited faces computed using SH
    non_visited_face_colors = (
        SH2RGB(sugar._sh_coordinates_dc[:, 0]).clamp(0.0, 1.0).view(n_triangles, -1, 3)
    )
    non_visited_face_colors = non_visited_face_colors.mean(dim=1)

    # Base color and roughness for non-visited faces
    non_visited_face_base_colors = (
        ((torch.sigmoid(sugar._features_base_color) *  0.77 + 0.03)
        * torch.ones(3, dtype=torch.float, device="cuda")[None, :])
    )
    non_visited_face_base_colors = torch.where(
        non_visited_face_base_colors <= 0.0031308,
        12.92 * non_visited_face_base_colors,
        1.055 * torch.pow(non_visited_face_base_colors, 1.0/2.4) - 0.055
    ).clamp(0.0, 1.0).view(n_triangles, -1, 3)

    non_visited_face_base_colors = non_visited_face_base_colors.mean(dim=1)
    
    non_visited_face_roughness = (
        (torch.sigmoid(sugar._features_roughness) * 0.9 + 0.09)
        .clamp(0.0, 1.0).view(n_triangles, -1, 1)
    )
    non_visited_face_roughness = non_visited_face_roughness.mean(dim=1)

    # Build rasterizer
    height = sugar.nerfmodel.training_cameras.gs_cameras[0].image_height
    width = sugar.nerfmodel.training_cameras.gs_cameras[0].image_width
    raster_settings = RasterizationSettings(image_size=(height, width))
    rasterizer = MeshRasterizer(
        cameras=sugar.nerfmodel.training_cameras,
        raster_settings=raster_settings,
        use_nvdiffrast=True,
    )

    print(f"Processing images...")
    for cam_idx in range(len(sugar.nerfmodel.training_cameras)):
        if texture_with_gaussian_renders:
            # Render color image
            images = sugar.render_image_gaussian_rasterizer(
                nerf_cameras=sugar.nerfmodel.training_cameras,
                camera_indices=cam_idx,
                sh_deg=n_sh,
                return_2d_radii=True,
                return_opacities=True,
                return_colors=True,
                bg_color=torch.tensor(bg_color, device=device),
                compute_color_in_rasterizer=True,
            )

            # Render color image
            rgb_img = images["image"].nan_to_num().clamp(min=0, max=1)
            rgb_img = rgb_img.view(1, height, width, 3)
            
            base_color_img = images["base_color"].nan_to_num().clamp(min=0, max=1)
            base_color_img = base_color_img.view(1, height, width, 3)
            
            roughness_img = images["roughness"].nan_to_num().clamp(min=0, max=1)
            roughness_img = roughness_img.view(1, height, width, 1)

            save_dir = os.path.join("output_renders", f"cam_{cam_idx:03d}")
            os.makedirs(save_dir, exist_ok=True)

            # Save RGB image
            rgb_np = (rgb_img[0].cpu().numpy() * 255).astype(np.uint8)
            imageio.imwrite(os.path.join(save_dir, "rgb.png"), rgb_np)

            # Save base color
            base_color_np = (base_color_img[0].cpu().numpy() * 255).astype(np.uint8)
            imageio.imwrite(os.path.join(save_dir, "base_color.png"), base_color_np)

            # Save roughness (convert from 1-channel to 3-channel grayscale for better visualization)
            roughness_np = (roughness_img.repeat(1, 1, 1, 3)[0].cpu().numpy() * 255).astype(np.uint8)
            imageio.imwrite(os.path.join(save_dir, "roughness.png"), roughness_np)

            fragments = rasterizer(sugar.surface_mesh, cam_idx=cam_idx)
            bary_coords = fragments.bary_coords.view(1, height, width, 3)
            pix_to_face = fragments.pix_to_face.view(1, height, width)

            mask = pix_to_face > -1
            face_indices = pix_to_face[mask]
            bary_coords = bary_coords[mask]
            
            colors = rgb_img[mask]
            base_colors = base_color_img[mask]
            roughness = roughness_img[mask]

            # Update face color statistics
            face_count[face_indices] = face_count[face_indices] + 1
            face_colors[face_indices] = face_colors[face_indices] + colors
            face_base_colors[face_indices] = (
                face_base_colors[face_indices] + base_colors
            )
            face_roughness[face_indices] = face_roughness[face_indices] + roughness

            # Map to texture coordinates and update texture images
            pixel_idx_0 = (
                (verts_uv[faces_uv[face_indices]] * bary_coords[:, :, None]).sum(dim=1)
                * texture_size
            ).int()

            # Update main texture
            texture_img[pixel_idx_0[:, 1], pixel_idx_0[:, 0]] = (
                texture_img[pixel_idx_0[:, 1], pixel_idx_0[:, 0]] + colors
            )
            texture_count[pixel_idx_0[:, 1], pixel_idx_0[:, 0]] = (
                texture_count[pixel_idx_0[:, 1], pixel_idx_0[:, 0]] + 1
            )
            texture_base_color[pixel_idx_0[:, 1], pixel_idx_0[:, 0]] = (
                texture_base_color[pixel_idx_0[:, 1], pixel_idx_0[:, 0]] + base_colors
            )
            texture_roughness[pixel_idx_0[:, 1], pixel_idx_0[:, 0]] = (
                texture_roughness[pixel_idx_0[:, 1], pixel_idx_0[:, 0]] + roughness
            )

        else:
            raise NotImplementedError(
                "Should use GT RGB image if texture_with_gaussian_renders is False."
            )

    # For visited UV points, we just average the colors from the rendered images
    filled_mask = texture_count[..., 0] > 0
    texture_img[filled_mask] = texture_img[filled_mask] / texture_count[filled_mask]
    texture_base_color[filled_mask] = (
        texture_base_color[filled_mask] / texture_count[filled_mask]
    )
    texture_roughness[filled_mask] = (
        texture_roughness[filled_mask] / texture_count[filled_mask]
    )

    # For non visited UV points belonging to visited faces, we use the average color of the face
    visited_faces_mask = face_count[..., 0] > 0
    face_colors[visited_faces_mask] = (
        face_colors[visited_faces_mask] / face_count[visited_faces_mask]
    )
    face_base_colors[visited_faces_mask] = (
        face_base_colors[visited_faces_mask] / face_count[visited_faces_mask]
    )
    face_roughness[visited_faces_mask] = (
        face_roughness[visited_faces_mask] / face_count[visited_faces_mask]
    )

    # For non visited UV points belonging to non visited faces, we use the averaged SH color
    face_colors[~visited_faces_mask] = non_visited_face_colors[~visited_faces_mask]
    face_base_colors[~visited_faces_mask] = non_visited_face_base_colors[~visited_faces_mask]
    face_roughness[~visited_faces_mask] = non_visited_face_roughness[~visited_faces_mask]

    # We fill the unvisited UV points with the corresponding face color
    texture_img[~filled_mask] = face_colors[uv_to_faces[~filled_mask]]
    texture_base_color[~filled_mask] = face_base_colors[uv_to_faces[~filled_mask]]
    texture_roughness[~filled_mask] = face_roughness[uv_to_faces[~filled_mask]]

    texture_img = texture_img.flip(0)

    # Create combined texture with color (RGB), base_color (RGB) and roughness (R) in 7 channels
    # For simplicity, we'll use the standard RGB texture as the main output
    # In a real workflow, you might want to save these additional textures separately

    # Return the textured mesh
    textures_uv = TexturesUV(
        maps=texture_img[None].float(),
        verts_uvs=verts_uv[None],
        faces_uvs=faces_uv[None],
        sampling_mode="nearest",
    )
    textured_mesh = Meshes(
        verts=[surface_mesh.verts_list()[0]],
        faces=[surface_mesh.faces_list()[0]],
        textures=textures_uv,
    )
    textured_mesh.texture_base_color = texture_base_color.flip(0)
    textured_mesh.texture_roughness = texture_roughness.flip(0)

    return textured_mesh
