import gradio as gr
import requests
import base64
import json
import io
from PIL import Image

def encode_image_to_base64(image):
    """Convert a PIL Image to base64 string"""
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str

def chat_with_llama(message, history, image=None):
    """Send message and optional image to Llama 3.2 Vision via Ollama API"""
    
    # Define the API endpoint for Ollama
    api_url = "http://localhost:11434/api/generate"
    
    # Prepare the messages from history and current message
    messages = []
    
    # Add history
    for human_msg, assistant_msg in history:
        messages.append({"role": "user", "content": human_msg})
        messages.append({"role": "assistant", "content": assistant_msg})
    
    # Create current message content
    current_content = []
    
    # Add text content
    current_content.append({
        "type": "text",
        "text": message
    })
    
    # Add image if provided
    if image is not None:
        # Convert image to PIL Image if it's not already
        if not isinstance(image, Image.Image):
            image = Image.open(image)
        
        # Encode image to base64
        img_base64 = encode_image_to_base64(image)
        
        # Add image content
        current_content.append({
            "type": "image",
            "image": img_base64
        })
    
    # Add current message
    messages.append({
        "role": "user",
        "content": current_content
    })
    
    # Create the request payload
    payload = {
        "model": "llama3.2-vision:11b-instruct-fp16",
        "messages": messages,
        "stream": False
    }
    
    try:
        # Send the request to Ollama API
        response = requests.post(api_url, json=payload)
        response.raise_for_status()
        
        # Parse the response
        result = response.json()
        return result.get("response", "No response received")
    
    except requests.exceptions.RequestException as e:
        return f"Error communicating with Ollama: {str(e)}"

with gr.Blocks(title="Chat with Llama 3.2 Vision") as demo:
    gr.Markdown("# Chat with Llama 3.2 Vision")
    gr.Markdown("Upload an image and chat with Llama 3.2 Vision model running on Ollama.")
    
    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(height=500)
            message = gr.Textbox(placeholder="Type your message here...", label="Message")
            with gr.Row():
                submit_btn = gr.Button("Send", variant="primary")
                clear_btn = gr.Button("Clear")
        
        with gr.Column(scale=1):
            image_input = gr.Image(label="Upload Image (Optional)", type="pil")
            image_info = gr.Markdown("Supported formats: JPG, JPEG, PNG")
    
    def user_input(message, history, image):
        if not message and image is None:
            return "", history, None
        
        # Add user message to history
        history.append((message, ""))
        
        return "", history, image
    
    def bot_response(history, image):
        if not history:
            return history
        
        # Get the last user message
        last_message = history[-1][0]
        
        # Get response from Llama
        response = chat_with_llama(last_message, history[:-1], image)
        
        # Update the last history entry with the bot's response
        history[-1] = (last_message, response)
        
        return history
    
    def clear_conversation():
        return [], None
    
    # Set up the interaction
    msg_state = gr.State()
    img_state = gr.State()
    
    submit_btn.click(
        user_input,
        inputs=[message, chatbot, image_input],
        outputs=[message, chatbot, img_state]
    ).then(
        bot_response,
        inputs=[chatbot, img_state],
        outputs=[chatbot]
    )
    
    message.submit(
        user_input,
        inputs=[message, chatbot, image_input],
        outputs=[message, chatbot, img_state]
    ).then(
        bot_response,
        inputs=[chatbot, img_state],
        outputs=[chatbot]
    )
    
    clear_btn.click(clear_conversation, outputs=[chatbot, image_input])

if __name__ == "__main__":
    demo.launch(share=True)
