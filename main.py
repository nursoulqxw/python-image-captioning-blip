from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import gradio as gr

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")


def generate_caption(img):
    img_input = Image.fromarray(img, mode="RGB")
    inputs = processor(img_input, return_tensors="pt") # preprocess the image, pt = pytorch tensor
    out = model.generate(**inputs) # generate caption
    caption = processor.decode(out[0], skip_special_tokens=True) # decode the output
    return caption

demo = gr.Interface(fn=generate_caption, 
                    inputs=[gr.Image(label="Image")]
                    ,outputs=[gr.Text(label="Caption"),],)

demo.launch()