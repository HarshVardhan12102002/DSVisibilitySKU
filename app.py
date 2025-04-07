import gradio as gr
import pandas as pd
import requests

sku_list = []

def identify_skus(image):
    response = requests.post("https://your-api.com/identify", files={"image": image})
    data = response.json()  # Expecting {"skus": ["Chilli Powder 100g", "Turmeric 200g", ...]}
    global sku_list
    sku_list = [{"SKU": sku, "Count": 0} for sku in data["skus"]]
    return pd.DataFrame(sku_list)

def count_boxes(top_image):
    response = requests.post("https://your-api.com/count", files={"image": top_image})
    counts = response.json()["counts"]  # Expecting {"Chilli Powder 100g": 5, ...}
    for row in sku_list:
        row["Count"] = counts.get(row["SKU"], 0)
    return pd.DataFrame(sku_list)

def export_csv(dataframe):
    csv = dataframe.to_csv(index=False)
    return csv

with gr.Blocks() as demo:
    with gr.Row():
        input_image = gr.Image(type="filepath", label="Upload Front Image")
        identify_button = gr.Button("Identify SKUs")
    
    sku_table = gr.Dataframe(headers=["SKU", "Count"], interactive=True)
    
    with gr.Row():
        top_view = gr.Image(type="filepath", label="Upload Top-View Image")
        count_button = gr.Button("Auto Count Boxes")
    
    export_btn = gr.Button("Export to CSV")
    csv_file = gr.File(label="Download CSV")

    identify_button.click(identify_skus, inputs=input_image, outputs=sku_table)
    count_button.click(count_boxes, inputs=top_view, outputs=sku_table)
    export_btn.click(export_csv, inputs=sku_table, outputs=csv_file)

demo.launch()
