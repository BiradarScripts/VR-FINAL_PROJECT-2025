# 🧠📷 Multimodal Visual Question Answering with ABO Dataset

---

## 🗂️ Project Overview

This project leverages the extensive **Amazon Berkeley Objects (ABO)** dataset to build a powerful **Multimodal Visual Question Answering (VQA)** system.

### 🔍 Objectives:
- 🎯 **Dataset Augmentation**: Create a multiple-choice VQA dataset from ABO using **Low-Rank Adaptation (LoRA)**.
- 🔧 **Baseline Modeling**: Establish robust baseline models and systematically evaluate them.
- 📈 **Performance Boost**: Fine-tune models to significantly enhance their accuracy and generalization.
- 🤖 **End Goal**: Develop a high-performing model capable of understanding visual data and accurately answering related questions.

---

## 📁 Dataset Resources

Here are the essential datasets and resources used in the project:

| 📌 Dataset / Model | 🔗 Link |
|--------------------|--------|
| 📦 ABO Dataset | [View on Kaggle](https://www.kaggle.com/datasets/aryamanpathak/abo-dataset) |
| 🧠 Master Train (JSON) | [View on Kaggle](https://www.kaggle.com/datasets/biradar1913/master-train) |
| 🧪 Master Test (JSON) | [View on Kaggle](https://www.kaggle.com/datasets/biradar1913/master-test) |
| 📊 Master Train (CSV) | [View on Kaggle](https://www.kaggle.com/datasets/biradar1913/master-traincsv) |
| 📊 Master Test (CSV) | [View on Kaggle](https://www.kaggle.com/datasets/biradar1913/master-testcsv) |
| 🤖 BLIP Fine-Tuned Model Versions | [View on Kaggle](https://www.kaggle.com/datasets/biradar1913/blip-finetunedmodel-versions) |

---

## 🤗 Model on Hugging Face

Our Visual Question Answering model is hosted on Hugging Face:  
🔗 [View the model on Hugging Face](https://huggingface.co/aryamanpathak/blip-vqa-abo)

---
📦 Report(Please dowload for better view of all 46 pages)  
🔗 [View Report](https://github.com/BiradarScripts/VR-FINAL_PROJECT-2025/blob/main/Vr_report.pdf)
---


## 📂 Folder Structure

```bash
└── biradarscripts-vr-final_project-2025/
    ├── README.md
    ├── BaseLineEvaluation/                                  
    │   ├── baseline-inference-master-test.ipynb            #Final inference script for blip-vqa-base 350m
    │   ├── Test-inference.ipynb                            #test inference script for blip-vqa-base/blip-2(2.7b)/blip-2(3b)/blip-2(11b)/Bakllava/Qwen models
    │   └── Result/                                           
    │       └── BaselineEvaluation.txt                      #a .txt file containing the evaluation results for all 15 standard metrics assessed on the BLIP-VQA-base model
    ├── DataCuration/                                      
    │   ├── FinalGeneratedDataSummary/                    
    │   │   ├── category_counts.txt                         #A .txt file showing the detailed analysis of the final generated dataset based on its product_type
    │   │   ├── image_id_to_product_type.json               #A.json file showing the mapping between product_type and image_id's
    │   │   └── plots/                                      #This folder contains plots/analysis of 12 groups obtained.
    │   ├── MainCode/                                      
    │   │   ├── api_key.txt                                 #A file form where api-keys are extracted 
    │   │   ├── final.py                                    #the final data generation script
    │   │   └── prompt.txt                                  #high quality prompt
    │   ├── OriginalDataSummary/                            
    │   │   ├── product_type_analysis.txt                   #A .txt file showing the detailed analysis of the original dataset based on its product_type
    │   │   └── Plots/                                      #This folder contains plots/analysis of 12 groups of initial dataset.
    │   ├── SubCodes/                                       #Folder containing all the codes used during data proccessing 
    │   │   ├── analysis.py                                 #script to map product_types to image_id
    │   │   ├── distributionProdcutTypeFolder.py            #script to distribute the listing files into all the 576 sub-folders/product_types
    │   │   ├── filteringFields.py                          #script to filter the fields of the listings .json files.removes the non en_US language_tage fields and mixes all_image_id and other_image_id
    │   │   ├── image_finder.py                             #script to find image using image_id
    │   │   ├── imagetypeMapping.py                         #script to map product_types to image_id
    │   │   ├── jsonFormatters.py                           #script that formats the output from the gemini to json object 
    │   │   ├── mainScriptTest.py                           #a sub-script to test the main script
    │   │   ├── organized.py                                #a script to divide the dataset into train and test set
    │   │   ├── partition.py                                # a script that divides the sampled product_types into 3 parts.    
    │   │   ├── primaryFiltering.py                         # a script to remove the unneccesary fields from metadata
    │   │   └── sampling.py                                 #Proportional Tiered Sampling Algorithm  script
    │   └── Train-Test-Split/
    ├── EvaluationMetrics/
    │   ├── evaluation-script/
    │   │   └── final-evaluation-script.ipynb               #Final Evaluation Script integrated with 15 evaluation metric to do inference on all the 13 blip-finetuned versions
    │   └── Results/                                        #A folder containing .txt files containing the evaluation results for all 15 standard metrics assessed on all the 13 finetuned blip versions
    │       ├── evaluation_metric_v1.txt
    │       ├── evaluation_metric_v10.txt
    │       ├── evaluation_metric_v11.txt
    │       ├── evaluation_metric_v12.txt
    │       ├── evaluation_metric_v13.txt
    │       ├── evaluation_metric_v2.txt
    │       ├── evaluation_metric_v3.txt
    │       ├── evaluation_metric_v4.txt
    │       ├── evaluation_metric_v5.txt
    │       ├── evaluation_metric_v6.txt
    │       ├── evaluation_metric_v7.txt
    │       ├── evaluation_metric_v8.txt
    │       └── evaluation_metric_v9.txt
    ├── FineTuningLora/                                    #A folder containing all 13 .ipynb scripts used for training and testing of the 13 models
    │   ├── blip-final-v1-train-test-2.ipynb
    │   ├── blip-final-v10-train-test.ipynb
    │   ├── blip-final-v12-train-test.ipynb
    │   ├── blip-final-v13-train-test.ipynb
    │   ├── blip-final-v2-train-test.ipynb
    │   ├── blip-final-v3-train-test.ipynb
    │   ├── blip-final-v4-train-test.ipynb
    │   ├── blip-final-v5-train-tests.ipynb
    │   ├── blip-final-v6-train-test.ipynb
    │   ├── blip-final-v7-train-test.ipynb
    │   ├── blip-final-v8-train-test.ipynb
    │   └── blip-final-v9-train-test.ipynb
    └── IMT2022529/                                         #submission folder.
        ├── inference.py
        └── requirements.txt
```


## 📁 Dataset Setup

1. **Download the ABO dataset**  
   Make sure to download the [abo-images-small.tar](https://amazon-berkeley-objects.s3.amazonaws.com/index.html) and place it in the **root directory** of this project.

---

## 🧹 Data Curation Instructions

### 📂 Navigate to:

```bash
cd DataCuration/MainCode
````

### 🔑 Setup API Key:

* Create a file named `api_key.txt` and place all your Gemini API keys inside it.
* Create a `.env` file in the same directory with the following content:

```env
GEMINI_API_KEY="PUT YOUR API KEY HERE"
```

### ▶️ Run the final data curation script:

```bash
python final.py
```

---

## 🤖 Inference Instructions

### 📂 Navigate to:

```bash
cd IMT2022529
```

### 📦 Install dependencies:

```bash
pip install -r requirements.txt
```
---
➕ ADD DATA

To add custom data for inference:

Go to the /data directory.
Drop all your images into this folder.
Open the metadata.csv file and fill in the following columns for each image:
image_name – The exact name of the image file (e.g., item01.jpg)
question – A relevant question for the image
answer – The expected or annotated answer
Example row in metadata.csv:

```bash
image_name,question,answer
1.jpg,What is the color of the object?,Red
```
---
### ▶️ Run inference:

```bash
python inference.py --image_dir /path/to/data --csv_path /path/to/data/metadata.csv
```

Replace `/path/to/...` with the appropriate path on your system. For example:

```bash
python inference.py --image_dir /Users/biradar/Documents/sem6/inference-setup/data --csv_path /Users/biradar/Documents/sem6/inference-setup/data/metadata.csv
```

---

## 📓 Running Jupyter Notebooks

To run any `.ipynb` notebook:

* Open the notebook in [Kaggle](https://www.kaggle.com/).
* Add the necessary dataset links as inputs.
* Run the cells as usual.

---

## 🧪 Using BLIP Fine-Tuned Model Versions

When using fine-tuned BLIP model versions, update the corresponding model paths to:

```
/v_x
```

Where `x` is the version number of the model you want to use.

---


##  Team Members

| Name              | Student ID    
| :---------------- | :------------ 
| Aryaman Pathak    | IMT2022513 
| Rutul Patel       | IMT2022021   
| Shreyas Biradar   | IMT2022529   

---

##  Acknowledgments

This project is undertaken as part of the **Visual Recognition** course at **International Institute of Information Technology, Bangalore**. We extend our gratitude to the course instructors and teaching assistants for their guidance and the opportunity to work on this exciting project.

---

