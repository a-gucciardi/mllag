from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification
import torch, random
import gradio as gr
import base64
import time
import os
# import google.generativeai as genai


### TEST
### Pour faire une interface web sexy

# Classification model
image_processor = AutoImageProcessor.from_pretrained("./swin_tiny_tuned")
model = AutoModelForImageClassification.from_pretrained("./swin_tiny_tuned")


# Create the Model
# txt_model = genai.GenerativeModel('gemini-pro')
# vis_model = genai.GenerativeModel('gemini-pro-vision')

# Image to Base 64 Converter
def image_to_base64(image_path):
    with open(image_path, 'rb') as img:
        encoded_string = base64.b64encode(img.read())
    return encoded_string.decode('utf-8')

# Function that takes User Inputs and displays it on ChatUI
def query_message(history,txt,img):
    if not img:
        history += [(txt,None)]
        return history
    base64 = image_to_base64(img)
    data_url = f"data:image/jpeg;base64,{base64}"
    history += [(f"{txt} ![]({data_url})", None)]
    return history

def classifier(image_path):
    # read image
    image = Image.open(image_path)
    encoding = image_processor(image.convert("RGB"), return_tensors="pt")

    # forward pass
    with torch.no_grad():
        outputs = model(**encoding)
        logits = outputs.logits

    predicted_class_idx = logits.argmax().item()
    dic_class = {0:"glioma", 1:"meningioma", 2:"no", 3:"pituitary"}
    print("True class name:", result:=dic_class[predicted_class_idx])

    return result

# Function that takes User Inputs, generates Response and displays on Chat UI
def llm_response(history, text, img):
    # img - img path
    if not img:
        response = "Hi"
        history += [(None,response.text)]
        return history
    
    if not text:
        response = f"This is an image with {classifier(img)} tumor."
        history += [(None,response)]
        return history

    else:
        img = Image.open(img)
        response = "Hi 2"
        history += [(None,response.text)]
        return history

# Interface Code
with gr.Blocks() as app:
    with gr.Row():
        image_box = gr.Image(type="filepath")
    
        chatbot = gr.Chatbot(
            scale = 2,
            height=750
        )
    text_box = gr.Textbox(
            placeholder="Enter text and press enter, or upload an image",
            container=False,
        )

    btn = gr.Button("Submit")
    clicked = btn.click(query_message,
                        [chatbot,text_box,image_box],
                        chatbot
                        ).then(llm_response,
                                [chatbot,text_box,image_box],
                                chatbot
                                )
app.queue()
app.launch(debug=True, share = True)