import sys

import gradio as gr
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification, pipeline
from PIL import Image
import numpy as np

# Global variables to store model and processor
model = None
image_processor = None
classifier_pipeline = None


def load_model(model_name=""):
    """Load the fine-tuned model and image processor"""
    global model, image_processor, classifier_pipeline

    try:
        # Load the image processor
        image_processor = AutoImageProcessor.from_pretrained(model_name)

        # Load the model
        model = AutoModelForImageClassification.from_pretrained(model_name)

        # Create a pipeline for easier inference
        classifier_pipeline = pipeline("image-classification", model=model_name)

        return f"✅ Model '{model_name}' loaded successfully!"
    except Exception as e:
        return f"❌ Error loading model: {str(e)}"


def classify_image_pipeline(image):
    """Classify image using the pipeline approach (simpler)"""
    if classifier_pipeline is None:
        return "❌ Please load a model first!"

    try:
        # Use the pipeline for classification
        results = classifier_pipeline(image)

        # Format results for display
        formatted_results = []
        for i, result in enumerate(results[:5]):  # Show top 5 predictions
            label = result["label"].replace("_", " ").title()
            score = result["score"]
            confidence = f"{score * 100:.2f}%"
            formatted_results.append(f"{i+1}. {label}: {confidence}")

        return "\n".join(formatted_results)

    except Exception as e:
        return f"❌ Error during classification: {str(e)}"


def classify_image_manual(image):
    """Classify image using manual approach (as shown in the document)"""
    if model is None or image_processor is None:
        return "❌ Please load a model first!"

    try:
        # Preprocess the image
        inputs = image_processor(image, return_tensors="pt")

        # Get predictions
        with torch.no_grad():
            logits = model(**inputs).logits

        # Get probabilities
        probabilities = torch.nn.functional.softmax(logits, dim=-1)

        # Get top 5 predictions
        top5_prob, top5_indices = torch.topk(probabilities, 5)

        # Format results
        formatted_results = []
        for i in range(5):
            idx = top5_indices[0][i].item()
            prob = top5_prob[0][i].item()
            label = model.config.id2label[idx].replace("_", " ").title()
            confidence = f"{prob * 100:.2f}%"
            formatted_results.append(f"{i+1}. {label}: {confidence}")

        # Also get the single top prediction
        predicted_label = logits.argmax(-1).item()
        top_prediction = (
            model.config.id2label[predicted_label].replace("_", " ").title()
        )

        result_text = (
            f"🏆 Top Prediction: {top_prediction}\n\n📊 All Top 5 Predictions:\n"
            + "\n".join(formatted_results)
        )
        return result_text

    except Exception as e:
        return f"❌ Error during classification: {str(e)}"

load_model(sys.argv[1])
def create_demo():
    """Create the Gradio interface"""

    with gr.Blocks(
        title="Document Classifier",
        theme=gr.themes.Soft(),
        css="""
        .gradio-container {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        }
        .container {
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(10px);
            border-radius: 15px;
            padding: 20px;
        }
        """,
    ) as demo:

        gr.Markdown(
            """
            Upload an image of invoices and get AI-powered classification results!
            
            This demo uses a Vision Transformer (ViT) model fine-tuned on the RVl DCIP dataset
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 🔧 Model Setup")
                # model_name = gr.Textbox(
                #     label="Model Name",
                #     value="nateraw/food",  # Default food classification model
                #     placeholder="Enter HuggingFace model name (e.g., 'your-username/food-classifier')",
                #     info="Enter the name of your fine-tuned model or use the default",
                # )
                # load_btn = gr.Button("Load Model", variant="primary")
                # model_status = gr.Textbox(
                #     label="Model Status", value="No model loaded", interactive=False
                # )

            with gr.Column(scale=2):
                gr.Markdown("### 📸 Image Classification")
                image_input = gr.Image(
                    label="Upload Image", type="pil", height=300
                )

                with gr.Row():
                    classify_pipeline_btn = gr.Button(
                        "🚀 Classify (Pipeline)", variant="primary"
                    )

                results_output = gr.Textbox(
                    label="Classification Results", lines=8, max_lines=10
                )

        # Example images section
        gr.Markdown("### 🖼️ Try These Example Images")
        gr.Examples(
            examples=[
                [
                    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/beignets-task-guide.png"
                ],
            ],
            inputs=image_input,
            label="Click to load example",
        )

        # Event handlers
        # load_btn.click(fn=load_model, inputs=model_name, outputs=model_status)

        classify_pipeline_btn.click(
            fn=classify_image_pipeline, inputs=image_input, outputs=results_output
        )

        # classify_manual_btn.click(
        #     fn=classify_image_manual, inputs=image_input, outputs=results_output
        # )

    return demo


if __name__ == "__main__":
    # Create and launch the demo
    demo = create_demo()

    # Launch with public sharing enabled (set share=False for local only)
    demo.launch(
        share=True,  # Set to False if you don't want public sharing
        server_name="0.0.0.0",  # Allow access from any IP
        server_port=7860,  # Default Gradio port
        show_error=True,
    )