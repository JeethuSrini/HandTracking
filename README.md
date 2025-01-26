# Hand Tracking Pipeline

This project implements a robust hand tracking pipeline using **MediaPipe HandLandmarker** for detecting hand landmarks and **SAM2 (Segment Anything Model 2)** for precise segmentation and tracking of hand regions across video frames. MediaPipe provides initial hand keypoints, which are used by SAM2 to refine and propagate segmentation masks, ensuring accurate and reliable hand tracking. The pipeline processes video inputs using **OpenCV** for frame extraction, leverages **PyTorch** for deep learning computations (supporting GPU/CPU), and outputs a video with segmented hand regions highlighted using green overlays. This implementation is ideal for applications such as gesture recognition, sign language interpretation, and other hand-tracking-related tasks.

## Video Demo

Below is a demonstration of the hand tracking pipeline in action. The video shows how the pipeline processes input video frames, detects hand landmarks, refines segmentation masks, and outputs the final video with highlighted hand regions.


Watch the demo video [here](https://drive.google.com/file/d/1NhKWTdGmIXRuuJFRPUTKTe9AI5u_fuJi/view?usp=sharing).

## Instructions for Setting Up the Environment

To set up the required environment, follow these steps(ensure that your in HandTracking directory(this directory)):

1. Create a new Conda environment named `handtracking`:
   ```bash
   conda create -n handtracking python=3.11 -y
   ```

2. Activate the environment:
   ```bash
   conda activate handtracking
    ```
3. Install required libraries for Sam2:
   ```bash
   git clone https://github.com/JeethuSrini/HandTracking.git && cd HandTracking
   pip install -e .
   ```

If you are installing on Windows, it's strongly recommended to use [Windows Subsystem for Linux (WSL)](https://learn.microsoft.com/en-us/windows/wsl/install) with Ubuntu.

## Getting Started

### Download Checkpoints

4. First, we need to download a model checkpoint. All the model checkpoints can be downloaded by running:

   ```bash
   cd checkpoints && \
   ./download_ckpts.sh && \
   cd ..
   ```

or individually from:

- [sam2.1_hiera_tiny.pt](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt)
- [sam2.1_hiera_small.pt](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt)
- [sam2.1_hiera_base_plus.pt](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt)
- [sam2.1_hiera_large.pt](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt)

(note that these are the improved checkpoints denoted as SAM 2.1; see [Model Description](#model-description) for details.)


6. Move to second sam2 dir:
    ```bash
    cd sam2
    ```

5. Installing the requirements:
    ```bash
    pip install -r requirements.txt
    ```

6. Run the script:
    ```bash
    python pipeline.py
    ```



