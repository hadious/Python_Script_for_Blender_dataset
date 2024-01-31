# Blender Custom Object 2D-Depthmap Dataset

This project aims to generate a custom object 2D-depthmap dataset using Blender, a powerful open-source 3D creation software. The dataset is intended for various computer vision and machine learning applications, such as object recognition, depth estimation, and scene understanding.

## Overview

Blender offers extensive capabilities for scene creation, object manipulation, and rendering, making it a suitable tool for generating synthetic datasets. In this project, we leverage Blender's functionalities to create custom objects and capture their corresponding depth maps.
## Dataset Generation Process

    Object Creation: Custom objects are designed and modeled within Blender, allowing for flexibility in shape, size, and texture.

    Scene Setup: The scene is configured to position the custom objects within a suitable environment. Lighting, camera angles, and other scene elements are adjusted to ensure realistic rendering.

    Rendering: Blender's rendering engine is utilized to produce high-quality images of the scene, including RGB images and depth maps.

    Depth Map Capture: Depth maps are captured using Blender's compositing features. A specific rendering pipeline is employed to ensure accurate depth representation.

## Depth Map Capture Procedure

To capture the depth map in the compositing section of Blender, the render layer should be connected to the normalize layer and a viewer. For detailed instructions, please refer to the following [Link](https://www.saifkhichi.com/blog/blender-depth-map-surface-normals).

## Dataset Applications

The generated dataset can be utilized for various purposes, including:

    Training and evaluating depth estimation algorithms
    Testing object recognition and segmentation models
    Enhancing scene understanding in computer vision tasks

## Dataset Usage

Researchers, developers, and enthusiasts are encouraged to explore and utilize the dataset for academic, research, and non-commercial projects. Contributions and feedback are welcome to improve the dataset and its applicability across different domains.
## Contributors

    [M.Hadi Sepanj]
    [University of Waterloo]
