# Language model for report generation project

In this project, image description and report generation abilities of language models are evaluated. The
aim is to link medical images and simple labels with text retrieval, increasing the usability and ex-
plainability of existing classification datasets. The system first uses a simple classifier, which allows the
retrieval of the image label, and then the label is used to create a pseudo-medical report. The goal is to
evaluate the llm’s abilities at data comprehension. 

## Content
**Report Generation**
In this folder, all the scripts and logs for the report generation are saved.  
Notebooks provide details on the workflow and usage : 
 - [1: classifier train](./report_generation/1_classifier_train.ipynb): training process of the Swin classifier for caption generation from labels.
 - [2: pipeline](./report_generation/2_pipeline.ipynb): complete pipeline usage of the report generation.
 - [3: results](./report_generation/3_results.ipynb): short analysis of the results obtained over the whole dataset used.

[mri_report3.json](./report_generation/mri_reports3.json) : latest version of the complete output, with index, caption and report generation. 


## WebUI 

*WIP* goal : create a simple Gradio webapp for quick report generation. 

