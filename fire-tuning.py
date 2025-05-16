import streamlit as st
import os
import yaml
from ultralytics import YOLO
import time
from datetime import datetime

st.set_page_config(page_title="YOLOv8 Fine-tuning", layout="wide")

st.title("YOLOv8 Fine-tuning Tool")

# Create two columns for inputs
col1, col2 = st.columns(2)

with col1:
    st.subheader("Dataset Settings")
    # Dataset path input (should contain train, test, valid folders)
    dataset_path = st.text_input("Dataset Path (containing train, test, valid folders)",
                                placeholder="/path/to/your/dataset")

    # Output path for trained weights
    output_path = st.text_input("Output Path for Trained Weights",
                                placeholder="/path/to/save/weights",
                                value="runs/train")

with col2:
    st.subheader("Training Settings")
    # Weights file input
    weights_path = st.text_input("YOLOv8 Weights Path (.pt file)",
                                placeholder="/path/to/yolo8n.pt",
                                value="yolo8n.pt")

    # Number of classes
    num_classes = 3  # Adjusted to match your dataset

    # Training epochs
    epochs = st.number_input("Training Epochs", min_value=1, value=50)

    # Batch size
    batch_size = st.number_input("Batch Size", min_value=1, value=16)

    # Image size
    img_size = st.number_input("Image Size", min_value=32, value=640)

# Advanced options in expander
with st.expander("Advanced Training Options"):
    patience = st.number_input("Early Stopping Patience", min_value=0, value=20)
    lr0 = st.number_input("Initial Learning Rate", min_value=0.0, value=0.01, format="%.4f")
    lrf = st.number_input("Final Learning Rate Factor", min_value=0.0, value=0.01, format="%.4f")
    momentum = st.number_input("SGD Momentum", min_value=0.0, max_value=1.0, value=0.937)
    weight_decay = st.number_input("Weight Decay", min_value=0.0, value=0.0005, format="%.5f")

# Function to validate paths
def validate_paths():
    errors = []

    # Check if dataset path exists
    if not os.path.exists(dataset_path):
        errors.append(f"Dataset path '{dataset_path}' does not exist!")
    else:
        # Check for train, test, valid folders
        for folder in ['train', 'test', 'valid']:
            full_path = os.path.join(dataset_path, folder)
            if not os.path.exists(full_path):
                errors.append(f"'{folder}' folder not found in dataset path!")
            else:
                # Check for images and labels subfolders
                for subfolder in ['images', 'labels']:
                    subfolder_path = os.path.join(full_path, subfolder)
                    if not os.path.exists(subfolder_path):
                        errors.append(f"'{folder}/{subfolder}' folder not found!")

    # Check if weights file exists
    if not os.path.exists(weights_path):
        errors.append(f"Weights file '{weights_path}' does not exist!")

    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)

    return errors

# Function to create YAML configuration for the dataset
def create_dataset_yaml():
    yaml_path = os.path.join(dataset_path, "dataset.yaml")

    # Create class names (placeholders)
    class_names = ["Fire", "default", "smoke"]

    dataset_config = {
        'path': dataset_path,
        'train': 'train/images',
        'val': 'valid/images',
        'test': 'test/images',
        'nc': num_classes,
        'names': class_names
    }

    with open(yaml_path, 'w') as f:
        yaml.dump(dataset_config, f, default_flow_style=False)

    return yaml_path

# Function to fine-tune the model
def finetune_model(yaml_path):
    # Initialize model
    model = YOLO(weights_path)

    # Start training
    results = model.train(
        data=yaml_path,
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        patience=patience,
        lr0=lr0,
        lrf=lrf,
        momentum=momentum,
        weight_decay=weight_decay,
        name=f"yolov8_finetune_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )

    # Get the path of the best model
    best_model_path = model.export()
    return best_model_path, results

# Start fine-tuning when button is clicked
if st.button("Start Fine-tuning"):
    # Validate paths
    errors = validate_paths()

    if errors:
        for error in errors:
            st.error(error)
    else:
        # Create progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()

        # Create dataset YAML
        status_text.text("Creating dataset configuration...")
        yaml_path = create_dataset_yaml()

        # Start fine-tuning
        status_text.text("Starting fine-tuning process...")

        try:
            with st.spinner("Fine-tuning in progress... This may take a while."):
                best_model_path, results = finetune_model(yaml_path)

            st.success(f"Fine-tuning completed! Best model saved at: {best_model_path}")

            # Display training results
            st.subheader("Training Results")
            st.write(f"Best mAP50-95: {results.results_dict.get('metrics/mAP50-95(B)', 0):.4f}")
            st.write(f"Best mAP50: {results.results_dict.get('metrics/mAP50(B)', 0):.4f}")

            # Option to download the model
            st.download_button(
                label="Download Fine-tuned Model",
                data=open(best_model_path, "rb").read(),
                file_name=os.path.basename(best_model_path),
                mime="application/octet-stream"
            )

        except Exception as e:
            st.error(f"An error occurred during training: {str(e)}")

# Display instructions
st.markdown("---")
st.subheader("Instructions")
st.markdown("""
1. Enter the path to your dataset containing train, test, and valid folders
2. Each folder should have 'images' and 'labels' subfolders
3. Enter the path to your YOLOv8 weights file (e.g., yolo8n.pt)
4. Adjust training parameters as needed
5. Click 'Start Fine-tuning' to begin the process
6. Once complete, you can download the fine-tuned model
""")

# Show dataset structure example
st.markdown("**Expected Dataset Structure:**")
st.code("""
dataset/
├── train/
│   ├── images/
│   │   ├── img1.jpg
│   │   └── img2.jpg
│   └── labels/
│       ├── img1.txt
│       └── img2.txt
├── valid/
│   ├── images/
│   │   └── ...
│   └── labels/
│       └── ...
└── test/
    ├── images/
    │   └── ...
    └── labels/
        └── ...
""")
