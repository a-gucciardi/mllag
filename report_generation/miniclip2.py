import torch
import json
import csv
from datasets import load_dataset
from transformers import AutoModel, AutoTokenizer
from torchvision.transforms.functional import pil_to_tensor
from torchmetrics.multimodal.clip_score import CLIPScore
from transformers import BlipProcessor, BlipForImageTextRetrieval

# description model
model = AutoModel.from_pretrained('openbmb/MiniCPM-Llama3-V-2_5', trust_remote_code=True, torch_dtype=torch.float16, resume_download=True)
model = model.to(device='cuda')
# tokenizer
tokenizer = AutoTokenizer.from_pretrained('openbmb/MiniCPM-Llama3-V-2_5', trust_remote_code=True, resume_download=True)
model.eval()

# blip model
processor = BlipProcessor.from_pretrained("Salesforce/blip-itm-base-coco", resume_download=True)
blipmodel = BlipForImageTextRetrieval.from_pretrained("Salesforce/blip-itm-base-coco", resume_download=True)

dataset = load_dataset("sartajbhuvaji/Brain-Tumor-Classification")

# Initialize a list to store all the reports
# all_reports = []

def generate_report(image, caption):
    question = f'Generate a report of the MRI image presented, use this description : {caption}.'
    msgs = [{'role': 'user', 'content': question}]
    report = model.chat(
        image=image,
        msgs=msgs,
        tokenizer=tokenizer,
        sampling=True,
        temperature=0.7,
        system_prompt='Medical report style, focus on image description and conditions.'
    )
    
    return report

def get_clip_score(image, report):

    metric = CLIPScore(model_name_or_path="openai/clip-vit-base-patch16")
    clip_score = metric(pil_to_tensor(image), report[:50])
    clip_score = clip_score.detach().item()

    return clip_score

def get_blip_score(image, report):

    inputs = processor(images=image, text=report[:50], return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = blipmodel(**inputs)
        blip_score = outputs.itm_score.softmax(dim=1)[:, 1]
    blip_score = blip_score.item()

    return blip_score


def main(input, output):
    # 
    all_reports = []
    with open(input, mode='r') as input_file:
        csvFile = csv.reader(input_file)
        next(csvFile, None)  # skip header

        for index, (image, lines) in enumerate(zip(dataset['Training']['image'], csvFile)):
            res = generate_report(image, lines[1])

            ## SCORES
            clip_score = get_clip_score(image, res)
            blip_score = get_blip_score(image, res)

            # Create a dictionary for this report
            report = {
                "Index": lines[0],
                "Description": lines[1],
                "Generated_Report": res,
                "CLIP_Score": clip_score,
                "BLIP_Score": blip_score
            }

            # Add the report to our list
            all_reports.append(report)

            # prints
            print(f"Index {lines[0]}")
            print(f"Description {res}")
            print(f"CLIP Score {clip_score}")
            print(f"BLIP Score {blip_score}")

    # Save all reports to a JSON file
    with open(output, 'w') as json_file:
        json.dump(all_reports, json_file, indent=4)

    print("All reports have been saved to mri_reports.json")

# # Reading the short descriptions
# with open('mri_descriptions.csv', mode='r') as input_file:
#     csvFile = csv.reader(input_file)
#     next(csvFile, None)  # skip header

#     for index, (image, lines) in enumerate(zip(dataset['Training']['image'], csvFile)):
#         question = f'Generate a report of the MRI image presented, use this description : {lines[1]}.'
#         msgs = [{'role': 'user', 'content': question}]
#         res = model.chat(
#             image=image,
#             msgs=msgs,
#             tokenizer=tokenizer,
#             sampling=True,
#             temperature=0.7,
#             system_prompt='Medical report style, focus on image description and conditions.'
#         )

#         ## CLIP SCORE
#         metric = CLIPScore(model_name_or_path="openai/clip-vit-base-patch16")
#         clip_score = metric(pil_to_tensor(image), res[:50])
#         clip_score = clip_score.detach().item()

#         ## BLIP SCORE
#         inputs = processor(images=image, text=res[:50], return_tensors="pt", padding=True)
#         with torch.no_grad():
#             outputs = blipmodel(**inputs)
#             blip_score = outputs.itm_score.softmax(dim=1)[:, 1]
#         blip_score = blip_score.item()

#         # Create a dictionary for this report
#         report = {
#             "Index": lines[0],
#             "Description": lines[1],
#             "Generated_Report": res,
#             "CLIP_Score": clip_score,
#             "BLIP_Score": blip_score
#         }

#         # Add the report to our list
#         all_reports.append(report)

#         # prints
#         print(f"Index {lines[0]}")
#         print(f"Description {res}")
#         print(f"CLIP Score {clip_score}")
#         print(f"BLIP Score {blip_score}")

# # Save all reports to a JSON file
# with open('mri_reports.json', 'w') as json_file:
#     json.dump(all_reports, json_file, indent=4)

# print("All reports have been saved to mri_reports.json")


if __name__ == "__main__":
    input_csv = 'mri_descriptions.csv'
    output_json = 'mri_reports3.json'
    main(input_csv, output_json)
