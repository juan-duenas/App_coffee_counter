import gradio as gr
import re
import config  # Ensures paths are set up correctly
from croppie import infer_2_app
from countgd import countgd, draw_boxes_on_image, draw_polygon_preview, apply_polygon_crop, update_summary


def get_count_from_croppie(pil_image):
    """
    Run Croppie (YOLOv5) inference and return the detection count.
    """
    if pil_image is None:
        return 0
    try:
        _, summary_string, _ = infer_2_app(pil_image)
        # Parse count from summary string like "Detected 42 berries..."
        match = re.search(r"Detected (\d+)", summary_string)
        if match:
            return int(match.group(1))
        return 0
    except Exception as e:
        print(f"Error in Croppie inference: {e}")
        return 0


def get_count_from_countgd(pil_image, object_label="coffee fruit"):
    """
    Run CountGD inference and return the detection count.
    """
    if pil_image is None:
        return 0
    try:
        _, count_string, _ = countgd(pil_image, object_label, [])
        # Parse count from string like "Detected 42 coffee fruit(s)"
        match = re.search(r"Detected (\d+)", count_string)
        if match:
            return int(match.group(1))
        return 0
    except Exception as e:
        print(f"Error in CountGD inference: {e}")
        return 0


def yield_prediction(image1, image2, image3, model_choice, object_label, productive_branches, num_trees):
    """
    Process three images through the selected model, calculate mean fruit count,
    and compute yield predictions.
    
    Returns:
        - results_text: Summary of yield predictions
        - counts_text: Individual image counts
    """
    images = [image1, image2, image3]
    valid_images = [img for img in images if img is not None]
    
    if len(valid_images) == 0:
        return "⚠️ Please upload at least one image.", ""
    
    # Get counts from selected model
    counts = []
    count_details = []
    
    for i, img in enumerate(images):
        if img is None:
            count_details.append(f"Image {i+1}: Not provided")
            continue
        
        if model_choice == "Croppie (YOLOv5)":
            count = get_count_from_croppie(img)
        else:  # CountGD
            label = object_label.strip() if object_label and object_label.strip() else "coffee fruit"
            count = get_count_from_countgd(img, label)
        
        counts.append(count)
        count_details.append(f"Image {i+1}: {count} fruits detected")
    
    if len(counts) == 0:
        return "⚠️ No valid images processed.", ""
    
    # Calculate average fruit count per branch (C)
    C = sum(counts) / len(counts)
    
    # Get productive branches (P) - default to 30.0 if not provided
    try:
        P = float(productive_branches) if productive_branches else 30.0
    except (ValueError, TypeError):
        P = 30.0
    
    # Get total trees (F) - None if not provided
    F = None
    if num_trees:
        try:
            F = int(num_trees)
        except (ValueError, TypeError):
            F = None
    
    # Calculate total fruit load per tree (T)
    T = P * C
    
    # Calculate total field yield (Y) if F is provided
    Y = T * F if F is not None else None
    
    # Build results text
    results = []
    results.append("## 🌿 Yield Prediction Results\n")
    results.append(f"**Average fruit per branch (C):** {C:.1f}")
    results.append(f"**Productive branches per tree (P):** {P:.1f}")
    results.append(f"**Total berry load per tree (T = P × C):** {T:.1f}")
    
    if Y is not None:
        results.append(f"**Total trees in field (F):** {F:,}")
        results.append(f"**Total estimated yield in field (Y = T × F):** {Y:,.0f} fruits")
    else:
        results.append("**Total trees in field (F):** Not provided")
        results.append("**Total estimated yield in field (Y):** N/A (provide tree count to calculate)")
    
    counts_text = "\n".join(count_details)
    counts_text += f"\n\n**Images analyzed:** {len(counts)}"
    
    return "\n\n".join(results), counts_text


with gr.Blocks() as demo:
    gr.Markdown(
    """
    # Coffee berries detection & counting
    This UI showcases two open source models that can estimate coffee yield based on images of branches with cherries on them. 
    Credits to the authors of each model can be found on the respective tabs.
    """
    )

    with gr.Tabs():
        # Tab 1: YOLOv5 Detector (custom trained model)
        with gr.Tab("Croppie"):
            gr.Markdown("""## Classic CNN approach.
                            This CNN was trained based on a YOLOv8 architecture and it was focused on Arabica Coffee (*Coffea arabica* L.) cultivars. Additional details of training are described in this [publication](https://doi.org/10.34133/plantphenomics.0165).
                            This demo uses a similar CNN trained on an earlier version of YOLO (v5). The checkpoints of that model are available [here](https://github.com/j-river1/Croppie)
                            .
                        """)
            with gr.Row():
                    with gr.Accordion("Instructions:"):
                                gr.Markdown("""
                                    1. Upload an input image using the interactive buttons or click on one of the examples below for automatic upload.
                                    2. Click on "Count Objects" to collect the results.      
                                    """)    
            with gr.Row():
                inp_yolo = gr.Image(label="Input image", type="pil")
                out_image_yolo = gr.Image(label="Annotated image")

            with gr.Row():
                out_summary = gr.Textbox(label="Detection summary")
                out_histogram = gr.Plot(label="Confidence Distribution")

            predict_button = gr.Button("Count Objects", variant="primary")

            predict_button.click(
                fn=infer_2_app,
                inputs=[inp_yolo],
                outputs=[out_image_yolo, out_summary, out_histogram]
            )

            # Get paths to example images
            example_image_paths = [
                str(p) for p in config.SAMPLE_IMAGES_DIR.glob("*") 
                if p.suffix.lower() in ['.jpg', '.jpeg', '.png']
            ]
            if example_image_paths:
                gr.Examples(
                    examples=example_image_paths,
                    inputs=inp_yolo
                )

        # Tab 2: CountGD with Walkthrough
        with gr.Tab("CountGD"):
            gr.Markdown("""## Open world object detection model with guided workflow.
                           Follow the step-by-step walkthrough to crop, annotate, and count objects.
                           The underlying model is reached via Gradio client. The original UI can be found [here](https://huggingface.co/spaces/nikigoli/countgd).
                        """)
            
            # State variables for walkthrough
            wt_click_points = gr.State([])
            wt_crop_coords = gr.State([])
            wt_working_image = gr.State(None)
            
            with gr.Accordion("Walkthrough Instructions", open=True):
                gr.Markdown("""
                    Follow the step-by-step workflow below to count objects in your image:
                    1. Upload an image and optionally crop to focus on a target area
                    2. Draw bounding boxes around example objects you want to count
                    3. Optionally add a text description of the target object
                    4. Submit to get counting results
                """)
            
            with gr.Walkthrough(selected=1) as walkthrough:
                
                # ============== STEP 1: Input Image & Crop ==============
                with gr.Step("Step 1: Input Image & Crop", id=1):
                    gr.Markdown("""### 📷 Upload and Clip Image
                    1. Upload an image from your device or select from examples below. 
                    2. Optionally, click multiple times on the image to define a polygon clip region.
                    3. Click at least 3 points to form a polygon. The region inside will be extracted.
                    4. Check preview of polygon on image to the right. 
                    5. Decide on the next actions with the buttons below: start again, clip, continue without clipped image.            
                    """)
                    
                    with gr.Row():
                        with gr.Column(scale=1):
                            wt_input_image_step1 = gr.Image(
                                label="Click to upload or select from examples", 
                                type="pil",
                                height=400
                            )
                            wt_crop_status = gr.Textbox(
                                label="Crop Status", 
                                value="Upload an image to begin",
                                interactive=False
                            )
                            
                        with gr.Column(scale=1):
                            wt_crop_preview = gr.Image(
                                label="Clip Preview (green polygon shows selected area)",
                                type="pil",
                                height=400,
                                interactive=False
                            )
                    
                    with gr.Row():
                        wt_clear_crop_btn = gr.Button("🔄 Clear Polygon", size="sm")
                        wt_close_polygon_btn = gr.Button("🔷 Close Polygon", variant="secondary")
                        wt_apply_crop_btn = gr.Button("✂️ Apply Clip", variant="secondary")
                        wt_next_step1_btn = gr.Button("Next: Draw Boxes →", variant="primary")
                    
                    wt_cropped_image_display = gr.Image(
                        label="Clipped Image (will be used in next steps)",
                        type="pil",
                        height=300,
                        visible=False
                    )
                    
                    if example_image_paths:
                        gr.Examples(
                            examples=example_image_paths,
                            inputs=wt_input_image_step1,
                            label="Example Images"
                        )
                
                # ============== STEP 2: Draw Bounding Boxes ==============
                with gr.Step("Step 2: Draw Visual Examples", id=2):
                    gr.Markdown("""### 🖱️ Draw Bounding Boxes
                    Click on the image to define bounding boxes around example objects.
                    **Each box requires 2 clicks** (any two diagonal corners).
                    Draw at least one box to help the model understand what to count.
                    """)
                    
                    with gr.Row():
                        with gr.Column(scale=1):
                            wt_box_input_image = gr.Image(
                                label="Click to draw boxes (2 clicks = 1 box)",
                                type="pil",
                                height=400
                            )
                            
                        with gr.Column(scale=1):
                            wt_box_preview_image = gr.Image(
                                label="Preview with drawn boxes",
                                type="pil",
                                height=400,
                                interactive=False
                            )
                    
                    wt_box_info_step2 = gr.Textbox(
                        label="Box Status",
                        value="Click on the image to start drawing boxes",
                        interactive=False
                    )
                    
                    with gr.Row():
                        wt_back_step2_btn = gr.Button("← Back to Crop")
                        wt_clear_boxes_step2_btn = gr.Button("🗑️ Clear All Boxes", variant="secondary")
                        wt_next_step2_btn = gr.Button("Next: Add Description →", variant="primary")
                
                # ============== STEP 3: Text Description ==============
                with gr.Step("Step 3: Object Description (Optional)", id=3):
                    gr.Markdown("""### 📝 Describe the Target Object
                    Optionally provide a text description of what you want to count.
                    This can improve accuracy, especially if visual examples are ambiguous.
                    
                    **Examples:** "coffee fruit", "ripe cherry", "green bean", "red fruit"
                    """)
                    
                    wt_object_label_input = gr.Textbox(
                        label="Object Description",
                        placeholder="e.g., coffee fruit, ripe cherry, bean...",
                        info="Leave empty to rely only on visual examples"
                    )
                    
                    with gr.Accordion("📋 Current Configuration Summary", open=True):
                        wt_config_summary = gr.Markdown("Configuration will appear here...")
                    
                    with gr.Row():
                        wt_back_step3_btn = gr.Button("← Back to Draw Boxes")
                        wt_next_step3_btn = gr.Button("Next: Submit & Count →", variant="primary")
                
                # ============== STEP 4: Submit and Results ==============
                with gr.Step("Step 4: Submit & Results", id=4):
                    gr.Markdown("""### 🚀 Submit for Object Counting
                    Review your configuration and click "Count Objects" to run the detection.
                    """)
                    
                    with gr.Row():
                        with gr.Column(scale=1):
                            wt_final_input_preview = gr.Image(
                                label="Input Image (with examples)",
                                type="pil",
                                height=300,
                                interactive=False
                            )
                            wt_final_text_display = gr.Textbox(
                                label="Object Description",
                                interactive=False
                            )
                            wt_final_boxes_display = gr.Textbox(
                                label="Visual Examples",
                                interactive=False
                            )
                        
                        with gr.Column(scale=1):
                            wt_output_image_result = gr.Image(
                                label="Detection Result",
                                type="pil",
                                height=300
                            )
                            wt_output_count_result = gr.Textbox(
                                label="Predicted Count"
                            )
                            wt_output_box_info = gr.Textbox(
                                label="Detection Details"
                            )
                    
                    with gr.Row():
                        wt_back_step4_btn = gr.Button("← Back to Description")
                        wt_submit_count_btn = gr.Button("🔍 Count Objects", variant="primary", size="lg")
                        wt_restart_btn = gr.Button("🔄 Start Over", variant="secondary")
            
            # ============== WALKTHROUGH EVENT HANDLERS ==============
            
            # --- Step 1 handlers ---
            def wt_handle_step1_click(image, state, evt: gr.SelectData):
                """Handle clicks on step 1 image for polygon selection."""
                if image is None:
                    return [], None, "No image loaded"
                
                x, y = evt.index[0], evt.index[1]
                new_state = state + [x, y]
                num_points = len(new_state) // 2
                preview = draw_polygon_preview(image, new_state, closed=False)
                
                if num_points == 1:
                    return new_state, preview, f"Point 1 at ({x}, {y}). Click to add more points (min 3 for polygon)."
                elif num_points == 2:
                    return new_state, preview, f"{num_points} points defined. Add at least 1 more point to form a polygon."
                else:
                    return new_state, preview, f"✓ {num_points} points defined. Add more points or click 'Close Polygon' to complete."
            
            def wt_close_polygon(image, coords):
                """Close the polygon and show preview."""
                if image is None:
                    return coords, None, "No image loaded"
                
                num_points = len(coords) // 2
                if num_points < 3:
                    return coords, draw_polygon_preview(image, coords, closed=False), f"Need at least 3 points. Currently have {num_points}."
                
                preview = draw_polygon_preview(image, coords, closed=True)
                return coords, preview, f"✓ Polygon closed with {num_points} points. Click 'Apply Clip' to extract region."
            
            def wt_clear_crop_selection(image):
                return [], image, "Polygon cleared. Click to define new points."
            
            def wt_apply_crop(image, coords):
                """Apply polygon clipping - wrapper for UI integration."""
                cropped, status, success = apply_polygon_crop(image, coords)
                return cropped, cropped, status, gr.update(visible=success)
            
            def wt_go_to_step2(image, cropped):
                final_image = cropped if cropped is not None else image
                if final_image is None:
                    return (gr.Walkthrough(selected=1), None, None, None, 
                            [], "Please upload an image first")
                return (gr.Walkthrough(selected=2), final_image, final_image, final_image, 
                        [], "Click on the image to define bounding boxes")
            
            def wt_reset_on_new_upload(image):
                if image is None:
                    return [], None, "Upload an image to begin"
                return [], image, "Image loaded. Click to define polygon points (optional)."
            
            # Connect Step 1 events
            wt_input_image_step1.select(
                fn=wt_handle_step1_click,
                inputs=[wt_input_image_step1, wt_crop_coords],
                outputs=[wt_crop_coords, wt_crop_preview, wt_crop_status]
            )
            
            wt_input_image_step1.change(
                fn=wt_reset_on_new_upload,
                inputs=[wt_input_image_step1],
                outputs=[wt_crop_coords, wt_crop_preview, wt_crop_status]
            )
            
            wt_clear_crop_btn.click(
                fn=wt_clear_crop_selection,
                inputs=[wt_input_image_step1],
                outputs=[wt_crop_coords, wt_crop_preview, wt_crop_status]
            )
            
            wt_close_polygon_btn.click(
                fn=wt_close_polygon,
                inputs=[wt_input_image_step1, wt_crop_coords],
                outputs=[wt_crop_coords, wt_crop_preview, wt_crop_status]
            )
            
            wt_apply_crop_btn.click(
                fn=wt_apply_crop,
                inputs=[wt_input_image_step1, wt_crop_coords],
                outputs=[wt_working_image, wt_cropped_image_display, wt_crop_status, wt_cropped_image_display]
            )
            
            wt_next_step1_btn.click(
                fn=wt_go_to_step2,
                inputs=[wt_input_image_step1, wt_working_image],
                outputs=[walkthrough, wt_working_image, wt_box_input_image, wt_box_preview_image, 
                        wt_click_points, wt_box_info_step2]
            )
            
            # --- Step 2 handlers ---
            def wt_handle_box_click(image, points, evt: gr.SelectData):
                if image is None:
                    return points, None, "No image available"
                
                x, y = evt.index[0], evt.index[1]
                new_points = points + [[x, y]]
                
                preview = draw_boxes_on_image(image, new_points)
                
                num_boxes = len(new_points) // 2
                pending = len(new_points) % 2
                
                if pending:
                    info = f"{num_boxes} box(es) complete. Click any diagonal corner to finish current box."
                else:
                    info = f"✓ {num_boxes} box(es) defined. Click to add more or proceed."
                
                return new_points, preview, info
            
            def wt_clear_all_boxes(image):
                return [], image, "All boxes cleared. Click to draw new boxes."
            
            def wt_go_back_to_step1():
                return gr.Walkthrough(selected=1)
            
            def wt_go_to_step3_fn(image, points, text):
                num_boxes = len(points) // 2
                status = f"{num_boxes} visual example(s) defined" if num_boxes > 0 else "No visual examples (text description recommended)"
                summary = update_summary(image, points, text)
                return gr.Walkthrough(selected=3), status, summary
            
            # Connect Step 2 events
            wt_box_input_image.select(
                fn=wt_handle_box_click,
                inputs=[wt_box_input_image, wt_click_points],
                outputs=[wt_click_points, wt_box_preview_image, wt_box_info_step2]
            )
            
            wt_clear_boxes_step2_btn.click(
                fn=wt_clear_all_boxes,
                inputs=[wt_box_input_image],
                outputs=[wt_click_points, wt_box_preview_image, wt_box_info_step2]
            )
            
            wt_back_step2_btn.click(
                fn=wt_go_back_to_step1,
                outputs=[walkthrough]
            )
            
            wt_next_step2_btn.click(
                fn=wt_go_to_step3_fn,
                inputs=[wt_working_image, wt_click_points, wt_object_label_input],
                outputs=[walkthrough, wt_box_info_step2, wt_config_summary]
            )
            
            # --- Step 3 handlers ---
            def wt_go_back_to_step2_fn(image):
                return gr.Walkthrough(selected=2), image, image
            
            def wt_go_to_step4_fn(image, points, text):
                summary = update_summary(image, points, text)
                num_boxes = len(points) // 2 if points else 0
                boxes_text = f"{num_boxes} bounding box(es)" if num_boxes > 0 else "No visual examples"
                text_display = text if text and text.strip() else "(not specified)"
                preview = draw_boxes_on_image(image, points) if image and points else image
                return gr.Walkthrough(selected=4), summary, preview, text_display, boxes_text
            
            # Connect Step 3 events
            wt_object_label_input.change(
                fn=update_summary,
                inputs=[wt_working_image, wt_click_points, wt_object_label_input],
                outputs=[wt_config_summary]
            )
            
            wt_back_step3_btn.click(
                fn=wt_go_back_to_step2_fn,
                inputs=[wt_working_image],
                outputs=[walkthrough, wt_box_input_image, wt_box_preview_image]
            )
            
            wt_next_step3_btn.click(
                fn=wt_go_to_step4_fn,
                inputs=[wt_working_image, wt_click_points, wt_object_label_input],
                outputs=[walkthrough, wt_config_summary, wt_final_input_preview, wt_final_text_display, wt_final_boxes_display]
            )
            
            # --- Step 4 handlers ---
            def wt_run_countgd(image, text, points):
                if image is None:
                    return None, "No image provided", "Please go back and upload an image"
                
                label = text.strip() if text and text.strip() else "object"
                return countgd(image, label, points)
            
            def wt_go_back_to_step3_fn():
                return gr.Walkthrough(selected=3)
            
            def wt_restart_workflow():
                return (
                    gr.Walkthrough(selected=1),
                    [],
                    [],
                    None,
                    None,
                    None,
                    "Upload an image to begin",
                    None,
                    "",
                    ""
                )
            
            # Connect Step 4 events
            wt_back_step4_btn.click(
                fn=wt_go_back_to_step3_fn,
                outputs=[walkthrough]
            )
            
            wt_submit_count_btn.click(
                fn=wt_run_countgd,
                inputs=[wt_working_image, wt_object_label_input, wt_click_points],
                outputs=[wt_output_image_result, wt_output_count_result, wt_output_box_info]
            )
            
            wt_restart_btn.click(
                fn=wt_restart_workflow,
                outputs=[
                    walkthrough, wt_click_points, wt_crop_coords, wt_working_image,
                    wt_input_image_step1, wt_crop_preview, wt_crop_status,
                    wt_output_image_result, wt_output_count_result, wt_output_box_info
                ]
            )
        
        # ============== Tab 3: Yield Prediction ==============
        with gr.Tab("Yield Prediction"):
            gr.Markdown("""## 🌿 Coffee Yield Estimation
            Upload three images of coffee branches to estimate yield based on fruit detection.
            The model will count fruits in each image and calculate:
            - **C**: Average number of fruits per branch
            - **T**: Total fruit load per tree (T = P × C)
            - **Y**: Total yield in the field (Y = T × F), if tree count is provided
            """)
            
            with gr.Accordion("Instructions:", open=True):
                gr.Markdown("""
                    1. Upload three images of coffee branches (or select from examples).
                    2. Choose the detection model to use.
                    3. Optionally provide:
                       - **Productive branches (P)**: Average number of productive branches per tree (default: 30.0)
                       - **Total trees (F)**: Number of trees in your field
                    4. Click "Calculate Yield" to get predictions.
                """)
            
            with gr.Row():
                model_selector = gr.Radio(
                    choices=["Croppie (YOLOv5)", "CountGD"],
                    value="Croppie (YOLOv5)",
                    label="Select Detection Model"
                )
                countgd_label = gr.Textbox(
                    label="Object Label (CountGD only)",
                    placeholder="e.g., coffee fruit",
                    value="coffee fruit",
                    visible=True
                )
            
            gr.Markdown("### 📷 Upload Three Branch Images")
            with gr.Row():
                yield_image1 = gr.Image(label="Image 1", type="pil", height=250)
                yield_image2 = gr.Image(label="Image 2", type="pil", height=250)
                yield_image3 = gr.Image(label="Image 3", type="pil", height=250)
            
            # Example images for easy selection
            if example_image_paths:
                gr.Markdown("**Example Images (click to load):**")
                with gr.Row():
                    gr.Examples(
                        examples=example_image_paths[:3] if len(example_image_paths) >= 3 else example_image_paths,
                        inputs=yield_image1,
                        label="Load to Image 1"
                    )
                with gr.Row():
                    gr.Examples(
                        examples=example_image_paths[:3] if len(example_image_paths) >= 3 else example_image_paths,
                        inputs=yield_image2,
                        label="Load to Image 2"
                    )
                with gr.Row():
                    gr.Examples(
                        examples=example_image_paths[:3] if len(example_image_paths) >= 3 else example_image_paths,
                        inputs=yield_image3,
                        label="Load to Image 3"
                    )
            
            gr.Markdown("### 🌳 Field Parameters")
            with gr.Row():
                productive_branches_input = gr.Number(
                    label="Productive branches per tree (P)",
                    value=30.0,
                    info="Average number of productive branches per tree. Default: 30.0",
                    precision=1
                )
                total_trees_input = gr.Number(
                    label="Total trees in field (F)",
                    value=None,
                    info="Optional: Total number of trees to calculate field yield",
                    precision=0
                )
            
            yield_button = gr.Button("🔍 Calculate Yield", variant="primary", size="lg")
            
            gr.Markdown("### 📊 Results")
            with gr.Row():
                with gr.Column(scale=1):
                    yield_results = gr.Markdown(label="Yield Predictions")
                with gr.Column(scale=1):
                    counts_output = gr.Textbox(label="Individual Image Counts", lines=6)
            
            yield_button.click(
                fn=yield_prediction,
                inputs=[
                    yield_image1, yield_image2, yield_image3,
                    model_selector, countgd_label,
                    productive_branches_input, total_trees_input
                ],
                outputs=[yield_results, counts_output]
            )

if __name__ == "__main__":
    demo.launch(share=False, allowed_paths=[str(config.SAMPLE_IMAGES_DIR)])