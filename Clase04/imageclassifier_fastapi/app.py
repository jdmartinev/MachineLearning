import gradio as gr
from fastai.learner import load_learner
from fastai.vision.core import PILImage


from huggingface_hub import hf_hub_download


learner = load_learner(hf_hub_download("jdmartinev/intel_image_classification_fastai","model.pkl"))

def classify_image(image):
    img = PILImage.create(image)
    pred_class, pred_idx, outputs = learner.predict(img)
    return(f"Predicted class: {pred_class}")
    
demo = gr.Interface(classify_image, gr.Image(), "text")
demo.launch(share=True)



