import os
# if using Apple MPS, fall back to CPU for unsupported ops
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python import BaseOptions
from sam2.build_sam import build_sam2_video_predictor


# Constants
NEGATIVE_FEEDBACK_POINTS = [[694, 68], [560, 27], [678, 27], [705, 103], [612, 75]]

# Configure device
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# if using Apple MPS, fall back to CPU for unsupported ops


if device.type == "cuda":
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
elif device.type == "mps":
        print(
            "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
            "give numerically different outputs and sometimes degraded performance on MPS. "
            "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
        )
# Paths
SAM2_CHECKPOINT = "../checkpoints/sam2.1_hiera_large.pt"
MODEL_CFG = "configs/sam2.1/sam2.1_hiera_l.yaml"
VIDEO_FILE = "test.mp4"
OUTPUT_DIR = "output_frames"
OUTPUT_VIDEO = "output_with_masks.mp4"


def hand_landmarker(frame_rgb):
    """Detect hand landmarks using MediaPipe HandLandmarker."""
    base_options = BaseOptions(model_asset_path='hand_landmarker.task')
    options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2)
    detector = vision.HandLandmarker.create_from_options(options)

    # Detect landmarks
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    detection_result = detector.detect(mp_image)

    pixel_coordinates = []
    if detection_result.hand_landmarks:
        for hand_landmarks in detection_result.hand_landmarks:
            for landmark in hand_landmarks:
                x_pixel = int(landmark.x * frame_rgb.shape[1])
                y_pixel = int(landmark.y * frame_rgb.shape[0])
                pixel_coordinates.append([x_pixel, y_pixel])
        print("Pixel coordinates:", pixel_coordinates)
    else:
        print("No hands detected.")
    return pixel_coordinates


def sam2_inference(points, frame_names, video_file):
    """Perform SAM2-based segmentation using input points."""
    predictor = build_sam2_video_predictor(MODEL_CFG, SAM2_CHECKPOINT, device=device)
    inference_state = predictor.init_state(video_path=video_file)
    predictor.reset_state(inference_state)

    ann_frame_idx = 0
    ann_obj_id = 1

    # Add negative feedback points
    points.extend(NEGATIVE_FEEDBACK_POINTS)
    points = np.array(points, dtype=np.float32)

    labels = np.array([1] * (len(points) - len(NEGATIVE_FEEDBACK_POINTS)) + [0] * len(NEGATIVE_FEEDBACK_POINTS), np.int32)

    _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=ann_frame_idx,
        obj_id=ann_obj_id,
        points=points,
        labels=labels,
    )

    # show the results on the current (interacted) frame
    plt.figure(figsize=(9, 6))
    plt.title(f"frame {ann_frame_idx}")
    plt.imshow(Image.open(os.path.join(video_file, frame_names[ann_frame_idx])))
    show_points(points, labels, plt.gca())
    show_mask((out_mask_logits[0] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_ids[0])
    plt.show()

    # Propagate segmentation across frames
    video_segments = {}
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }

    return video_segments


def save_video_with_masks(video_path, frame_names, video_segments, output_video_path, vis_frame_stride=1):
    """Save the output video with masked hands."""
    # Open the original video to get FPS and resolution
    video = cv2.VideoCapture(video_path)
    fps = int(video.get(cv2.CAP_PROP_FPS))
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video.release()

    # Initialize the VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Codec for .mp4
    out_video = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    for out_frame_idx in range(0, len(frame_names), vis_frame_stride):
        # Read the frame
        frame = cv2.imread(frame_names[out_frame_idx])
        if frame is None:
            continue

        # Overlay masks on the frame
        for out_obj_id, out_mask in video_segments.get(out_frame_idx, {}).items():
            mask = (out_mask > 0).astype(np.uint8) * 255  # Convert mask to binary
            color_mask = np.zeros_like(frame)
            color_mask[:, :, 1] = mask  # Apply mask in green channel

            # Blend the original frame with the mask
            alpha = 0.6  # Transparency factor
            frame = cv2.addWeighted(frame, 1 - alpha, color_mask, alpha, 0)

        # Write the frame to the output video
        out_video.write(frame)

    out_video.release()
    print(f"Output video saved at: {output_video_path}")

def show_mask(mask, ax, obj_id=None, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=200):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))    


def inference_frames(video_path, output_folder):
    """Extract frames from video and perform inference."""
    os.makedirs(output_folder, exist_ok=True)
    video = cv2.VideoCapture(video_path)

    if not video.isOpened():
        raise RuntimeError("Error: Could not open video.")

    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Total Frames: {total_frames}, FPS: {video.get(cv2.CAP_PROP_FPS)}")

    frame_names = []
    frame_index = 0
    points = []

    while True:
        ret, frame = video.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_name = f"{frame_index:04d}.jpg"
        frame_filename = os.path.join(output_folder, frame_name)
        cv2.imwrite(frame_filename, frame)
        # print(frame_filename)
        frame_names.append(frame_name)

        if frame_index == 0:
            points = hand_landmarker(frame_rgb)

        frame_index += 1

    video.release()
    print(OUTPUT_DIR)
    # Perform SAM2 inference
    video_segments = sam2_inference(points, frame_names, OUTPUT_DIR)

    # Save the output video
    save_video_with_masks(video_path, frame_names, video_segments, OUTPUT_VIDEO)


if __name__ == "__main__":
    inference_frames(VIDEO_FILE, OUTPUT_DIR)