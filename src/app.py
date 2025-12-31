import gradio as gr
import config  # Ensures paths are set up correctly
from croppie import infer_2_app
from countgd import countgd, draw_boxes_on_image
from PIL import ImageDraw


def update_summary(image, points, text):
    """Generate summary of current configuration."""
    lines = []
    
    if image is not None:
        w, h = image.size if hasattr(image, 'size') else ("?", "?")
        lines.append(f"- **Image**: Loaded ({w}x{h} pixels)")
    else:
        lines.append("- **Image**: ⚠️ Not loaded")
    
    num_boxes = len(points) // 2 if points else 0
    if num_boxes > 0:
        lines.append(f"- **Visual Examples**: {num_boxes} bounding box(es)")
    else:
        lines.append("- **Visual Examples**: None defined")
    
    if text and text.strip():
        lines.append(f"- **Text Description**: \"{text.strip()}\"")
    else:
        lines.append("- **Text Description**: Not provided")
    
    return "\n".join(lines)

with gr.Blocks() as demo:
    gr.Markdown(
    """
    # Coffee fuit Detection & Counting
    Compare two models for counting coffee fruits based on branch images.
    """
    )

    with gr.Tabs():
        # Tab 1: YOLOv5 Detector (custom trained model)
        with gr.Tab("Croppie"):
            gr.Markdown("""## Closed world object detection model.
                            The model was trained on images of Coffee tree branches. Architecture is based on YOLOv5.
                            This model only takes visual inputs as a reference and is not interactive.
                        """)
            with gr.Row():
                    with gr.Accordion("Instructions:"):
                                gr.Markdown("""
                                    1. Input an image using the interactive buttons or one available in the examples area.
                                    2. Click on "Count Objects" to collect the results.      
                                    """)    
            with gr.Row():
                inp_yolo = gr.Image(label="Input image", type="pil")
                out_image_yolo = gr.Image(label="Annotated image")

            out_summary = gr.Textbox(label="Detection summary")

            predict_button = gr.Button("Count Objects")

            predict_button.click(
                fn=infer_2_app,
                inputs=[inp_yolo],
                outputs=[out_image_yolo, out_summary]
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

        # Tab 2: CountGD Generic Counter (Plain interface)
        with gr.Tab("CountGD Plain"):
            gr.Markdown("""## Open world object detection model taking text (in English), visual exemplars, or both as prompts.
                           The underlying model is reached via Gradio client. The original UI can be found [here](https://huggingface.co/spaces/nikigoli/countgd).
                        """)
            
            # State to store click points
            click_points = gr.State([])
            
            with gr.Row():
                with gr.Accordion("Instructions:"):
                            gr.Markdown("""
                                1. Input an image using the interactive buttons or one available in the examples area.
                                2. Use your mouse to click on the input image to define vizual examples. 
                                3. Vizualise the user's examples in the preview image to the right.
                                4. Click "Clear Boxes" button if not satisfied.
                                5. Optionally input a text description of the target object.
                                6. Click on "Count Objects" to collect the results.      
                                """)
                            
            with gr.Row():
                inp_countgd = gr.Image(label="Click to select box corners (2 clicks = 1 box)", type="pil")
                preview_image = gr.Image(label="Preview with boxes", type="pil")                  
                    
            with gr.Row():
                object_label = gr.Textbox(
                label="Object to count", 
                placeholder="Optional description of target object; e.g. fruit"
                )
                
                with gr.Column():    
                    clear_boxes_button = gr.Button("Clear Boxes")
                    count_button = gr.Button("Count Objects", variant="primary")

            with gr.Row():
                out_image_countgd = gr.Image(label="Detected Instances")
                
                with gr.Column():
                    out_message = gr.Textbox(label="Predicted count")
                    box_info = gr.Textbox(label="Detection Boxes Info", interactive=False)
            
            def handle_click(image, points, evt: gr.SelectData):
                """Handle image clicks to build bounding boxes."""
                if image is None:
                    return points, None, "No boxes defined"
                
                x, y = evt.index[0], evt.index[1]
                new_points = points + [[x, y]]
                
                # Draw preview with current points/boxes
                preview = draw_boxes_on_image(image, new_points)
                
                num_boxes = len(new_points) // 2
                pending = len(new_points) % 2
                info = f"{num_boxes} box(es) defined"
                if pending:
                    info += ", 1 point pending (click again to complete box)"
                
                return new_points, preview, info
            
            def clear_boxes(image):
                """Clear all defined boxes."""
                return [], image, "Boxes cleared"
            
            def reset_on_new_image(image):
                """Reset points when a new image is uploaded."""
                return [], image, "Upload complete. Click to define boxes."

            inp_countgd.select(
                fn=handle_click,
                inputs=[inp_countgd, click_points],
                outputs=[click_points, preview_image, box_info]
            )
            
            inp_countgd.change(
                fn=reset_on_new_image,
                inputs=[inp_countgd],
                outputs=[click_points, preview_image, box_info]
            )
            
            clear_boxes_button.click(
                fn=clear_boxes,
                inputs=[inp_countgd],
                outputs=[click_points, preview_image, box_info]
            )

            count_button.click(
                fn=countgd,
                inputs=[inp_countgd, object_label, click_points],
                outputs=[out_image_countgd, out_message, box_info]
            )
            
            if example_image_paths:
                gr.Examples(
                    examples=example_image_paths,
                    inputs=inp_countgd
                )
        
        # Tab 3: CountGD with Walkthrough
        with gr.Tab("CountGD Walkthrough"):
            gr.Markdown("""## Open world object detection model with guided workflow.
                           Follow the step-by-step walkthrough to crop, annotate, and count objects.
                        """)
            
            # State variables for walkthrough
            wt_click_points = gr.State([])
            wt_crop_coords = gr.State([])
            wt_working_image = gr.State(None)
            
            with gr.Accordion("ℹ️ Walkthrough Instructions", open=True):
                gr.Markdown("""
                    Follow the step-by-step workflow below to count objects in your image:
                    - **Step 1**: Upload an image and optionally crop to focus on a target area
                    - **Step 2**: Draw bounding boxes around example objects you want to count
                    - **Step 3**: Optionally add a text description of the target object
                    - **Step 4**: Submit to get counting results
                """)
            
            with gr.Walkthrough(selected=1) as walkthrough:
                
                # ============== STEP 1: Input Image & Crop ==============
                with gr.Step("Step 1: Input Image & Crop", id=1):
                    gr.Markdown("""### 📷 Upload and Crop Image
                    Upload an image from your device or select from examples below. 
                    Optionally, click twice on the image to define a crop region (any two diagonal corners).
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
                                label="Crop Preview (green box shows selected area)",
                                type="pil",
                                height=400,
                                interactive=False
                            )
                    
                    with gr.Row():
                        wt_clear_crop_btn = gr.Button("🔄 Clear Crop Selection", size="sm")
                        wt_apply_crop_btn = gr.Button("✂️ Apply Crop", variant="secondary")
                        wt_next_step1_btn = gr.Button("Next: Draw Boxes →", variant="primary")
                    
                    wt_cropped_image_display = gr.Image(
                        label="Cropped Image (will be used in next steps)",
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
                """Handle clicks on step 1 image for crop selection."""
                if image is None:
                    return [], None, "No image loaded"
                
                x, y = evt.index[0], evt.index[1]
                new_state = state + [x, y]
                
                if len(new_state) > 4:
                    new_state = [x, y]
                
                preview = image.copy()
                draw = ImageDraw.Draw(preview)
                
                if len(new_state) == 2:
                    x1, y1 = new_state[0], new_state[1]
                    r = 8
                    draw.ellipse([x1 - r, y1 - r, x1 + r, y1 + r], fill="lime", outline="white", width=2)
                    return new_state, preview, f"First corner at ({x1}, {y1}). Click any diagonal corner."
                elif len(new_state) >= 4:
                    x1, y1, x2, y2 = new_state[0], new_state[1], new_state[2], new_state[3]
                    left, right = min(x1, x2), max(x1, x2)
                    top, bottom = min(y1, y2), max(y1, y2)
                    draw.rectangle([left, top, right, bottom], outline="lime", width=4)
                    return new_state, preview, f"✓ Crop area: ({left}, {top}) to ({right}, {bottom}). Click 'Apply Crop' or continue."
                
                return new_state, preview, "Click on the image to define crop area"
            
            def wt_clear_crop_selection(image):
                return [], image, "Crop selection cleared. Click to define new area."
            
            def wt_apply_crop(image, coords):
                if image is None:
                    return None, image, "No image to crop", gr.update(visible=False)
                
                if len(coords) < 4:
                    return image, image, "No crop area defined. Using full image.", gr.update(visible=False)
                
                x1, y1, x2, y2 = coords[0], coords[1], coords[2], coords[3]
                left, right = min(x1, x2), max(x1, x2)
                top, bottom = min(y1, y2), max(y1, y2)
                
                if right - left < 20 or bottom - top < 20:
                    return image, image, "Crop area too small. Using full image.", gr.update(visible=False)
                
                try:
                    cropped = image.crop((left, top, right, bottom))
                    return cropped, cropped, f"✓ Image cropped to ({left}, {top}) - ({right}, {bottom})", gr.update(visible=True)
                except Exception as e:
                    return image, image, f"Crop failed: {e}", gr.update(visible=False)
            
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
                return [], image, "Image loaded. Click to define crop area (optional)."
            
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
        
        #Tab 4 Original CountGD app using function load
        #with gr.Tab("Original CountGD"):
        #     gr.Markdown("""## Open world object detection model taking text (in English), visual exemplars, or both as prompts.
        #                """)
        #     gr.load("nikigoli/countgd", src="spaces")

if __name__ == "__main__":
    demo.launch(share=False, allowed_paths=[str(config.SAMPLE_IMAGES_DIR)])