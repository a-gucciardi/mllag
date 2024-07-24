import torch
from torch.utils.data import Dataset, DataLoader
from transformers import pipeline
from datasets import load_dataset
from tqdm import tqdm
import csv

dataset = load_dataset("sartajbhuvaji/Brain-Tumor-Classification")

model_id = "llava-hf/llava-1.5-7b-hf"
pipe = pipeline("image-to-text", model=model_id, device="cuda:2")

def descript(image, label):
    dic_class = {0:"glioma", 1:"meningioma", 2:"no", 3:"pituitary"}
    # image, label = example['image'], example['label']
        
    # if dic_class[label] != 2:
    #     prompt = f"USER: <image>\nGenerate a report of the displayed MRI, is it T1 or T2 weighted? In which area of the brain is located the tumor, \
    #             and can you locate where the {dic_class[label]} tumor is ? \
    #             Answer in the form of the sentence : This is a (T1 or T2 weighted) image of a brain with (your answer with the tumor name) tumor. It is located in (your answer).\
    #             \nASSISTANT:"
    # else: 
    #     prompt = f"USER: <image>\nGenerate a report of the displayed MRI, is it T1 or T2 weighted? In which area of the brain is located the tumor, \
    #             and can you confirm there is no tumor? \
    #             Answer in the form of the sentence : This is a (T1 or T2 weighted) image of a brain without tumor.\
    #             \nASSISTANT:"
    if dic_class[label] != 2:
        prompt = f"USER: <image>\nGenerate a report of the displayed MRI, is it T1 or T2 weighted? \
                Answer in the form of the sentence : This is a (T1 or T2 weighted) image of a brain with (your answer with the tumor name) tumor.\
                \nASSISTANT:"
    else: 
        prompt = f"USER: <image>\nGenerate a report of the displayed MRI, is it T1 or T2 weighted? \
                Answer in the form of the sentence : This is a (T1 or T2 weighted) image of a brain without tumor.\
                \nASSISTANT:"

    outputs = pipe(image, prompt=prompt, generate_kwargs={"max_new_tokens": 200})
    description = outputs[0]['generated_text'].split("ASSISTANT: ", maxsplit=1)[1]
    
    return description


def main():
    # Open a CSV file to write the descriptions
    with open('mri_descriptions2.csv', 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        
        # Write the header
        csvwriter.writerow(['Index', 'Description'])
        
        # Process each example in the training set
        for index, example in enumerate(tqdm(dataset['Training'])):
            description = descript(example['image'], example['label'])
            csvwriter.writerow([index, description])
            
            # Optional: print every 10th description to monitor progress
            if index % 10 == 0:
                print(f"Index {index}: {description}")

    print("Descriptions have been saved to mri_descriptions.csv")

if __name__ == "__main__":
    main()