import re
from weakref import ref
from differentiable_illum.utils import render
from render_util.disp_normal import DispNormal
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from datautil.visualizer import TensorVisualizer
from typing import Optional


class PinholeProjection:
    def __init__(self, location: torch.Tensor):
        self.location = location


class RenderInput:
    def __init__(
        self,
        image_nir: torch.Tensor,
        image_disp: torch.Tensor,
        image_occ: Optional[torch.Tensor],
        material_index: Optional[torch.Tensor],
        calibration_cam: torch.Tensor,
        baseline,
        projection: PinholeProjection,
    ):
        self.image_nir = image_nir
        self.image_disp = image_disp
        self.image_occ = image_occ
        self.calibration_cam = calibration_cam
        self.material_index = material_index
        self.baseline = baseline
        self.projection = projection


class RenderOption:
    def __init__(
        self,
        light_alpha: float = 0.99,
        ambient_alpha: float = 0.2,
        reflectance_alpha: float = 0.7,
    ):
        self.light_alpha = light_alpha
        self.ambient_alpha = ambient_alpha
        self.reflectance_alpha = reflectance_alpha


class Renderer2:
    def __init__(self, option: RenderOption = RenderOption()):
        self.disp_normal = DispNormal()
        self.option = option

    def shading_term(self, normal_map, projection_vector):
        return (normal_map * projection_vector).sum(dim=-1).clamp(min=0)

    def diffuse_reflectance(self, material_index_map):
        return material_index_map

    def light_propagation(self, light_source_position, position_vector):
        return (
            1
            / ((light_source_position - position_vector).norm(dim=-1) ** 2)
            * self.option.light_alpha
        )

    def render_image(self, input: RenderInput):
        # Compute Normal Map
        point_cloud = self.disp_normal.disp_to_pointcloud(
            input.image_disp, input.calibration_cam, input.baseline
        )

        normal_map = self.disp_normal.pointcloud_to_normal(point_cloud)

        # Compute Projection Vector

        projection_vector = input.projection.location - point_cloud
        projection_vector = F.normalize(projection_vector, dim=-1)

        # Compute Lamberitan Reflectance

        shading_term = self.shading_term(normal_map, projection_vector)
        if input.material_index is not None:
            reflectance = self.diffuse_reflectance(input.material_index)
        else:
            reflectance = torch.ones_like(input.image_nir)

        light = self.light_propagation(input.projection.location, point_cloud)
        lambertian_reflectance = reflectance * shading_term * light
        ambient = input.image_nir

        reflectance = reflectance / reflectance.max()

        rendered_image = (
            lambertian_reflectance * self.option.reflectance_alpha
            + ambient * self.option.ambient_alpha
        )

        rendered_image = (rendered_image.cpu().numpy() * 255).astype(np.uint8)

        # TensorVisualizer(normal_map.cpu().numpy()).plot()
        # cv2.imshow("disparity", input.image_disp.cpu().numpy().astype(np.uint8))
        cv2.imshow("Projected Vector", projection_vector.cpu().numpy())
        cv2.imshow("Rendered Image", rendered_image)

        cv2.imshow("Normal Map", normal_map.cpu().numpy())
        cv2.imshow("Nir Image", input.image_nir.cpu().numpy())
        cv2.imshow("lambda", lambertian_reflectance.cpu().numpy())
        cv2.waitKey(0)

        return rendered_image
