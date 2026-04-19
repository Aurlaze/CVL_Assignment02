import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import RectangleSelector
from PIL import Image

# ==========================================
# 1. HELPER: Intersection over Union
# ==========================================
def calculate_iou(box1, box2):
    """Calculates Intersection over Union (IoU)"""
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])

    overlap_area = max(0, x_right - x_left) * max(0, y_bottom - y_top)

    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    iou = overlap_area / float(box1_area + box2_area - overlap_area + 1e-6)
    return iou

# ==========================================
# 2. HELPER: Advanced Drag-and-Drop Annotation Tool
# ==========================================
def manual_annotation(image_path):
    """Opens an image so the user can drag a box to define Ground Truth."""
    print(f"    [?] Ground Truth box missing for '{image_path}'. Opening Annotation Tool...")
    img = Image.open(image_path)
    
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(img)
    ax.set_title(f"ANNOTATION MODE: {image_path}\n1. Click and DRAG to draw a box around the bulb.\n2. Adjust the corners if needed.\n3. CLOSE the window when finished.")
    
    # We use a list to store the coordinates so the callback function can update them
    box_coords = [0, 0, 10, 10]

    def onselect(eclick, erelease):
        """This function runs every time you finish dragging the box."""
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata
        
        # Save the current box coordinates
        box_coords[0] = int(min(x1, x2))
        box_coords[1] = int(min(y1, y2))
        box_coords[2] = int(max(x1, x2))
        box_coords[3] = int(max(y1, y2))
        print(f"    [*] Current box: {box_coords}", end='\r')

    # Create the interactive drag tool
    selector = RectangleSelector(
        ax, onselect,
        useblit=True,
        button=[1],  # Left mouse button only
        minspanx=5, minspany=5,
        spancoords='pixels',
        interactive=True # This allows you to grab the corners and resize the box!
    )
    
    # Pause the script until the user closes the image window
    plt.show()

    print(f"\n    [+] Final annotated coordinates saved: {box_coords}")
    return box_coords

# ==========================================
# 3. MAIN SCRIPT
# ==========================================
def main():
    # --- THE ALL-IN-ONE ANSWER KEY ---
    # Set to 'None' to trigger the manual clicker tool, or paste your coordinates here!
    ANSWER_KEY = {
        'traffic.jpg':  {'class': 'Red',    'box': None},
        'traffic2.jpg': {'class': 'Yellow', 'box': None},
        'traffic3.jpg': {'class': 'Green',  'box': None}
    }
    
    print(f"Starting pipeline for {len(ANSWER_KEY)} images...\n" + "="*40)

    # Loop through our dictionary
    for image_path, true_data in ANSWER_KEY.items():
        print(f"\n>>> Processing: '{image_path}'")
        
        if not os.path.exists(image_path):
            print(f"    [!] Error: Could not find '{image_path}'. Skipping.")
            continue 

        # --- A. Load Image ---
        img = np.array(Image.open(image_path).convert('RGB'))

        # --- B. Color Segmentation (The Algorithm's Guess) ---
        R, G, B = img[:, :, 0], img[:, :, 1], img[:, :, 2]
        
        red_mask = (R > 150) & (G < 100) & (B < 100)
        green_mask = (R < 100) & (G > 140) & (B < 150)
        yellow_mask = (R > 150) & (G > 150) & (B < 100)

        color_counts = {
            'Red': np.sum(red_mask),
            'Yellow': np.sum(yellow_mask),
            'Green': np.sum(green_mask)
        }
        
        predicted_class = max(color_counts, key=color_counts.get)
        max_pixels = color_counts[predicted_class]

        if max_pixels < 50:
            print("    [-] Not enough illuminated pixels detected. Skipping.")
            continue

        if predicted_class == 'Red':
            active_mask = red_mask
        elif predicted_class == 'Yellow':
            active_mask = yellow_mask
        else:
            active_mask = green_mask

        # Calculate Predicted Box based on the pixels found
        y_indices, x_indices = np.where(active_mask)
        predicted_box = [0, 0, 0, 0]
        if len(x_indices) > 0 and len(y_indices) > 0:
            predicted_box = [np.min(x_indices), np.min(y_indices), np.max(x_indices), np.max(y_indices)]
        
        # --- C. Get the Ground Truth (The Answer Key) ---
        ground_truth_class = true_data['class']
        ground_truth_box = true_data['box']

        # Trigger the manual annotation tool because box is None!
        if ground_truth_box is None:
            ground_truth_box = manual_annotation(image_path)

        # --- D. Calculate Assignment Metrics ---
        if predicted_class == ground_truth_class:
            classification_accuracy = 100.0
        else:
            classification_accuracy = 0.0

        iou_score = calculate_iou(ground_truth_box, predicted_box)

        # --- E. Visualization & Output ---
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        axes[0].imshow(active_mask * 255, cmap='gray')
        axes[0].set_title(f"Segmentation Mask: {predicted_class}")
        axes[0].axis('off')

        axes[1].imshow(img)
        axes[1].set_title("Assignment Analysis")
        axes[1].axis('off')

        # Draw Ground Truth Box (Thick Blue)
        rect_gt = patches.Rectangle((ground_truth_box[0], ground_truth_box[1]), 
                                    ground_truth_box[2]-ground_truth_box[0], ground_truth_box[3]-ground_truth_box[1], 
                                    linewidth=4, edgecolor='blue', facecolor='none', label='Ground Truth')
        axes[1].add_patch(rect_gt)

        # Draw Predicted Box (Thin Red)
        rect_pred = patches.Rectangle((predicted_box[0], predicted_box[1]), 
                                      predicted_box[2]-predicted_box[0], predicted_box[3]-predicted_box[1], 
                                      linewidth=2, edgecolor='red', facecolor='none', label='Prediction')
        axes[1].add_patch(rect_pred)
        
        axes[1].legend(loc='upper right')

        # DYNAMIC TEXT PLACEMENT: Move the box out of the way for traffic.jpg!
        if image_path == 'traffic.jpg':
            text_x, text_y = 0.98, 0.05  # Bottom-Right coordinates
            ha, va = 'right', 'bottom'
        else:
            text_x, text_y = 0.02, 0.95  # Top-Left coordinates
            ha, va = 'left', 'top'

        text_background = dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='gray')
        display_text = (f"--- METRICS ANALYSIS ---\n\n"
                        f"Predicted: {predicted_class}\n"
                        f"Ground Truth: {ground_truth_class}\n"
                        f"Accuracy: {classification_accuracy}%\n\n"
                        f"IoU (Overlap): {iou_score:.4f}")
        
        axes[1].text(text_x, text_y, display_text, transform=axes[1].transAxes, 
                     fontsize=11, horizontalalignment=ha, verticalalignment=va, bbox=text_background)

        plt.tight_layout()
        
        base_name = os.path.basename(image_path)           
        name_only = os.path.splitext(base_name)[0]         
        output_filename = f"{name_only}_evaluation.jpg"        
        
        fig.savefig(output_filename, bbox_inches='tight', dpi=300)
        plt.close(fig) 
        
        print(f"    [+] Saved final analysis to '{output_filename}'")

    print("\n" + "="*40 + "\nPipeline complete! Check your folder for the images.")

if __name__ == "__main__":
    main()