import streamlit as st
import os
import yaml
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
import pandas as pd
import time
import random
import cv2
from PIL import Image
import io
import sys
import traceback

st.set_page_config(page_title="YOLOv8 Model Validation", layout="wide")

st.title("YOLOv8 Model Validation Tool")

# Create two columns for inputs
col1, col2 = st.columns(2)

with col1:
    st.subheader("Model Settings")
    # Model path input
    model_path = st.text_input("YOLOv8 Model Path (.pt, .torchscript, etc.)", 
                             placeholder="/path/to/your/best.pt")
    
    # Dataset path or YAML file
    dataset_path = st.text_input("Dataset Path or YAML file", 
                               placeholder="/path/to/dataset.yaml or /path/to/dataset/")

with col2:
    st.subheader("Validation Settings")
    # Image size
    img_size = st.number_input("Image Size", min_value=32, value=640)
    
    # Confidence threshold
    conf_threshold = st.slider("Confidence Threshold", min_value=0.1, max_value=0.9, value=0.25, step=0.05)
    
    # IoU threshold for NMS
    iou_threshold = st.slider("IoU Threshold", min_value=0.1, max_value=0.9, value=0.7, step=0.05)

# Function to ensure dataset has a YAML file
def ensure_dataset_yaml(dataset_path):
    # If dataset_path is already a YAML file, return it
    if dataset_path.endswith('.yaml'):
        return dataset_path
    
    # Otherwise, look for a dataset.yaml in the path
    yaml_path = os.path.join(dataset_path, "dataset.yaml")
    if os.path.exists(yaml_path):
        return yaml_path
    
    # If no YAML file exists, try to create one
    st.warning("No dataset.yaml found. Attempting to create one...")
    
    # Check if train/valid/test structure exists
    if not all(os.path.exists(os.path.join(dataset_path, folder)) for folder in ['train', 'valid', 'test']):
        st.error("Dataset doesn't have the expected folder structure (train/valid/test)")
        return None
    
    # Guess number of classes based on label files
    nc = 0
    label_files = []
    valid_labels_dir = os.path.join(dataset_path, 'valid', 'labels')
    if os.path.exists(valid_labels_dir):
        label_files = [f for f in os.listdir(valid_labels_dir) if f.endswith('.txt')]
        
        if label_files:
            # Read the first few label files to guess number of classes
            for i, label_file in enumerate(label_files[:10]):
                with open(os.path.join(valid_labels_dir, label_file), 'r') as f:
                    for line in f:
                        if line.strip():
                            class_id = int(line.split()[0])
                            nc = max(nc, class_id + 1)
    
    if nc == 0:
        st.error("Could not determine number of classes. Please provide a YAML file.")
        return None
        
    # Create a basic YAML file
    dataset_config = {
        'path': dataset_path,
        'train': 'train/images',
        'val': 'valid/images',
        'test': 'test/images',
        'nc': nc,
        'names': [f"class{i}" for i in range(nc)]
    }
    
    yaml_path = os.path.join(dataset_path, "dataset.yaml")
    with open(yaml_path, 'w') as f:
        yaml.dump(dataset_config, f, default_flow_style=False)
    
    st.success(f"Created dataset.yaml with {nc} classes")
    return yaml_path

# Function to safely extract validation metrics
def extract_metrics(results):
    # Initialize with default values
    metrics = {
        'mAP50': 0.0,
        'mAP50-95': 0.0,
        'Precision': 0.0,
        'Recall': 0.0,
        'F1-Score': 0.0
    }
    
    # Try to extract metrics from validation results
    try:
        if hasattr(results, 'box'):
            if hasattr(results.box, 'map50'):
                metrics['mAP50'] = float(results.box.map50)
            if hasattr(results.box, 'map'):
                metrics['mAP50-95'] = float(results.box.map)
            if hasattr(results.box, 'p'):
                metrics['Precision'] = float(results.box.p)
            if hasattr(results.box, 'r'):
                metrics['Recall'] = float(results.box.r)
                
                # Calculate F1 only if both precision and recall are available
                p = float(results.box.p)
                r = float(results.box.r)
                metrics['F1-Score'] = 2 * (p * r) / (p + r + 1e-10)
    except Exception as e:
        st.warning(f"Could not extract some metrics: {str(e)}")
        
    return metrics

# Function to validate the model on the dataset
def validate_model(model, yaml_path, img_size, conf_threshold, iou_threshold):
    # Start validation
    start_time = time.time()
    results = model.val(
        data=yaml_path,
        imgsz=img_size,
        conf=conf_threshold,
        iou=iou_threshold,
        verbose=True
    )
    elapsed = time.time() - start_time
    
    # Safely extract metrics
    metrics = extract_metrics(results)
    
    return metrics, elapsed, results

# Function to visualize some validation samples
def visualize_samples(model, yaml_path, conf_threshold, iou_threshold, img_size, num_samples=5):
    # Load dataset info
    with open(yaml_path, 'r') as f:
        data_dict = yaml.safe_load(f)
    
    # Get validation image paths
    val_path = os.path.join(data_dict['path'], data_dict.get('val', 'valid/images'))
    
    # Handle potential differences in path structure
    if not os.path.exists(val_path):
        # Try alternatives
        alternatives = [
            os.path.join(data_dict['path'], 'valid/images'),
            os.path.join(data_dict['path'], 'valid'),
            os.path.join(data_dict['path'], 'val')
        ]
        
        for alt_path in alternatives:
            if os.path.exists(alt_path):
                val_path = alt_path
                break
    
    if not os.path.exists(val_path):
        st.error(f"Validation image path not found: {val_path}")
        return []
    
    # Get all image files
    val_images = []
    for root, _, files in os.walk(val_path):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                val_images.append(os.path.join(root, file))
    
    if not val_images:
        st.warning(f"No images found in validation path: {val_path}")
        return []
    
    # Select random samples
    num_samples = min(num_samples, len(val_images))
    samples = random.sample(val_images, num_samples)
    
    # Run predictions and store results
    results = []
    for img_path in samples:
        try:
            pred = model.predict(
                source=img_path,
                conf=conf_threshold,
                iou=iou_threshold,
                imgsz=img_size,
            )[0]
            
            # Get original image
            img = Image.open(img_path)
            
            # Create a copy of the image with predictions
            pred_img = pred.plot()
            pred_img = Image.fromarray(pred_img)
            
            # Get filename for display
            filename = os.path.basename(img_path)
            
            # Get detection info
            num_detections = len(pred.boxes)
            
            results.append({
                'filename': filename,
                'original_img': img,
                'pred_img': pred_img,
                'num_detections': num_detections
            })
        except Exception as e:
            st.warning(f"Error processing image {img_path}: {str(e)}")
    
    return results

# Button to start validation
if st.button("Start Validation"):
    if not os.path.exists(model_path):
        st.error(f"Model file not found: {model_path}")
    elif not os.path.exists(dataset_path):
        st.error(f"Dataset path not found: {dataset_path}")
    else:
        with st.spinner("Validating model..."):
            # Get YAML path
            yaml_path = ensure_dataset_yaml(dataset_path)
            
            if yaml_path:
                try:
                    # Show progress
                    progress_text = st.empty()
                    progress_text.text("Loading model...")
                    
                    # Load model
                    model = YOLO(model_path)
                    
                    # Create tabs for different validation views
                    tab1, tab2, tab3 = st.tabs(["Metrics", "Class Performance", "Sample Predictions"])
                    
                    # Run validation
                    progress_text.text("Running validation...")
                    metrics, elapsed_time, validation_results = validate_model(
                        model, yaml_path, img_size, conf_threshold, iou_threshold
                    )
                    
                    # Display metrics in the first tab
                    with tab1:
                        st.subheader("Model Performance Metrics")
                        
                        # Display metrics in a better format
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("mAP50", f"{metrics['mAP50']*100:.2f}%")
                            st.metric("Precision", f"{metrics['Precision']*100:.2f}%")
                        with col2:
                            st.metric("mAP50-95", f"{metrics['mAP50-95']*100:.2f}%")
                            st.metric("Recall", f"{metrics['Recall']*100:.2f}%")
                        with col3:
                            st.metric("F1-Score", f"{metrics['F1-Score']*100:.2f}%")
                            st.metric("Validation Time", f"{elapsed_time:.2f} seconds")
                            
                        # Interpret the results
                        st.subheader("Interpretation")
                        
                        map_score = metrics['mAP50']
                        if map_score > 0.75:
                            performance = "Excellent"
                            advice = "Your model is performing very well! It has high accuracy in detecting objects."
                        elif map_score > 0.5:
                            performance = "Good"
                            advice = "Your model is performing well, but there is room for improvement. Consider training for more epochs or collecting more diverse data."
                        elif map_score > 0.25:
                            performance = "Fair"
                            advice = "Your model has moderate performance. Try adjusting the learning rate, training longer, or improving the quality of your dataset."
                        else:
                            performance = "Needs Improvement"
                            advice = "Your model needs significant improvement. Consider checking your annotations, increasing dataset size, training for longer, or using a larger base model."
                        
                        st.write(f"**Performance Assessment**: {performance}")
                        st.write(advice)
                        
                        # Add download button for metrics
                        metrics_text = (
                            f"YOLOv8 Model Validation Results\n"
                            f"Model: {model_path}\n"
                            f"Dataset: {yaml_path}\n"
                            f"Image Size: {img_size}\n"
                            f"Confidence Threshold: {conf_threshold}\n"
                            f"IoU Threshold: {iou_threshold}\n\n"
                            f"Performance Metrics:\n"
                            f"mAP50: {metrics['mAP50']*100:.2f}%\n"
                            f"mAP50-95: {metrics['mAP50-95']*100:.2f}%\n"
                            f"Precision: {metrics['Precision']*100:.2f}%\n"
                            f"Recall: {metrics['Recall']*100:.2f}%\n"
                            f"F1-Score: {metrics['F1-Score']*100:.2f}%\n"
                            f"Validation Time: {elapsed_time:.2f} seconds\n\n"
                            f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}"
                        )
                        
                        st.download_button(
                            "Download Metrics Report",
                            metrics_text,
                            file_name="model_validation_report.txt",
                            mime="text/plain"
                        )
                    
                    # Display per-class metrics in the second tab
                    with tab2:
                        st.subheader("Per-Class Performance")
                        
                        # Load class names
                        with open(yaml_path, 'r') as f:
                            data_dict = yaml.safe_load(f)
                        class_names = data_dict.get('names', [f"class{i}" for i in range(data_dict.get('nc', 0))])
                        
                        # Display per-class metrics if available
                        try:
                            st.info("Attempting to extract per-class metrics...")
                            
                            # Safe approach to access class metrics
                            class_data = []
                            
                            # Try different ways to access class metrics based on YOLOv8 version
                            if hasattr(validation_results, 'box') and hasattr(validation_results.box, 'ap_class_index'):
                                # Method 1
                                class_indices = validation_results.box.ap_class_index
                                for i, idx in enumerate(class_indices):
                                    class_name = class_names[idx] if idx < len(class_names) else f"class{idx}"
                                    
                                    # Try to get AP values
                                    ap50 = validation_results.box.ap50[i] if hasattr(validation_results.box, 'ap50') and i < len(validation_results.box.ap50) else 0
                                    
                                    class_data.append({
                                        'Class': class_name,
                                        'AP50': f"{float(ap50)*100:.2f}%",
                                        'Index': int(idx)
                                    })
                            
                            # If no class data was found, show a message
                            if not class_data:
                                st.warning("Per-class metrics are not available in the validation results.")
                                st.info("This is common for YOLOv8 versions or when using specific model formats.")
                            else:
                                st.dataframe(pd.DataFrame(class_data))
                                
                        except Exception as e:
                            st.warning(f"Could not display per-class metrics: {str(e)}")
                            st.info("This typically happens with some YOLOv8 versions or export formats.")
                    
                    # Display sample predictions in the third tab
                    with tab3:
                        st.subheader("Sample Predictions")
                        
                        # Number of samples to show
                        num_samples = st.slider("Number of samples to display", min_value=1, max_value=10, value=5)
                        
                        # Visualize samples
                        progress_text.text("Generating sample predictions...")
                        sample_results = visualize_samples(
                            model, yaml_path, conf_threshold, iou_threshold, img_size, num_samples
                        )
                        
                        if not sample_results:
                            st.warning("No sample predictions could be generated. Please check your validation images.")
                        else:
                            # Display results in a grid (2 per row)
                            for i in range(0, len(sample_results), 2):
                                cols = st.columns(2)
                                for j in range(2):
                                    if i + j < len(sample_results):
                                        result = sample_results[i + j]
                                        with cols[j]:
                                            st.image(result['pred_img'], caption=f"{result['filename']} - {result['num_detections']} detections")
                                            with st.expander("View Original"):
                                                st.image(result['original_img'], caption=f"Original: {result['filename']}")
                        
                        progress_text.empty()
                
                except Exception as e:
                    st.error(f"An error occurred during validation: {str(e)}")
                    st.error(f"Details: {traceback.format_exc()}")

# Display instructions
st.markdown("---")
st.subheader("Instructions")
st.markdown("""
1. Enter the path to your trained YOLOv8 model
2. Enter the path to your dataset folder or YAML file
3. Adjust validation settings if needed
4. Click "Start Validation" to evaluate your model

The validation tool will:
- Calculate performance metrics (mAP, Precision, Recall, F1-Score)
- Attempt to show per-class performance (may not be available for all model formats)
- Visualize sample detections from your validation set
""")

# Troubleshooting section
with st.expander("Troubleshooting"):
    st.markdown("""
    **Common Issues:**
    
    1. **"Model file not found" error**
       - Make sure you entered the full correct path to your model file
       - Check that the file exists and has the correct extension (.pt, .torchscript, etc.)
    
    2. **"Dataset path not found" error**
       - Verify that the dataset folder exists and contains train/valid/test subfolders
       - Alternatively, provide a direct path to your dataset.yaml file
    
    3. **Per-class metrics not showing**
       - This is normal for some model formats (especially exported ones)
       - The basic metrics (mAP, precision, recall) should still be available
    
    4. **Sample predictions not showing**
       - Check that your validation folder contains image files
       - Try adjusting the confidence threshold to a lower value
    
    5. **Unsupported format string error**
       - This is a compatibility issue with certain model formats
       - The app is designed to handle this gracefully
    """)

