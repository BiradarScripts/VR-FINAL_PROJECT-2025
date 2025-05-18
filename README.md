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

Sure! Here's the **complete README.md code**, including all the text you mentioned, properly formatted in Markdown:



## 📁 Dataset Setup

1. **Download the ABO dataset**  
   Make sure to download the [ABO dataset](https://huggingface.co/datasets/StanfordVL/ABO) and place it in the **root directory** of this project.

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

This project is undertaken as part of the **[Course Name/Code, e.g., "Multimodal Machine Learning (CS XXXX)"]** course at **[Your University Name]**. We extend our gratitude to the course instructors and teaching assistants for their guidance and the opportunity to work on this exciting project.

---

##  License

This project is licensed under the **[Choose a License, e.g., MIT License]** - see the `LICENSE` file for details.

---
