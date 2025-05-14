# 🚀 Multimodal Visual Question Answering with Amazon Berkeley Objects (ABO) Dataset

## Project Overview

This project focuses on developing a **Multimodal Visual Question Answering (VQA)** system using the rich **Amazon Berkeley Objects (ABO)** dataset. Our objective is to create a multiple-choice VQA dataset from ABO, establish and evaluate robust baseline models, and then significantly improve their performance through fine-tuning using the **Low-Rank Adaptation (LoRA)** technique. The ultimate goal is to build a high-performing VQA model capable of answering questions about visual data, which will be rigorously evaluated on a hidden dataset and an undisclosed metric.

---

## 🎯 Introduction

Visual Question Answering (VQA) is a challenging multimodal task that requires models to understand both visual content (images) and natural language questions, then provide accurate answers. This project addresses the complexities of VQA by leveraging the diverse and detailed Amazon Berkeley Objects (ABO) dataset. We aim to create a practical multiple-choice VQA dataset, evaluate existing models, and innovate by applying LoRA for efficient and effective fine-tuning on resource-constrained environments.

---

## 🌟 Project Goals

Our primary objectives for this project are:

* **Dataset Generation:** Construct a comprehensive multiple-choice VQA dataset from the raw ABO data.
* **Baseline Establishment:** Evaluate the performance of established VQA models as baselines.
* **Efficient Fine-tuning:** Implement and apply **LoRA** for efficient fine-tuning of large pre-trained models.
* **Performance Optimization:** Achieve state-of-the-art or near state-of-the-art performance on the VQA task within given constraints.
* **Robust Evaluation:** Assess model performance using standard VQA metrics and prepare for evaluation on hidden test sets.

---

# 📦 Amazon Berkeley Objects (ABO) Dataset

The dataset comprises **147,702 product listings** enriched with **multilingual metadata** and a total of **398,212 unique catalog images**, making it a valuable resource for training and evaluating models in areas like product recognition, multilingual search, and e-commerce intelligence. A **smaller, more manageable variant** of the dataset is available for download (approximately **3GB** in size, compared to the original **100GB**), which includes product **metadata in CSV format** and **images resized to 256x256 pixels**. This compact version is ideal for quick experimentation and development without compromising on data quality. You can find the dataset [here](#) under the *Downloads* section—please select the "small variant" to avoid storage or processing issues.

Would you like this tailored for a research paper, README file, or project report?


## Dataset Overview: Amazon Berkeley Objects (ABO)

The Amazon Berkeley Objects (ABO) dataset is a comprehensive repository featuring diverse product images under various viewpoints and lighting conditions, alongside detailed metadata. It serves as an ideal foundation for training and evaluating advanced VQA models. For this project, two primary `.tar` files were essential:

### `abo-images-small.tar` (3 GB)

This archive contains downscaled (maximum 256 pixels) catalog images and their associated metadata. The downscaling is crucial for efficient storage, faster processing, and suitability as input for many deep learning models without significant loss of relevant visual information.

* **`abo-images-small/images/metadata/images.csv`**:
    * A gzip-compressed CSV file containing image metadata.
    * **Columns**:
        * `image_id`: Unique identifier for each image.
        * `height`: Original height of the image in pixels.
        * `width`: Original width of the image in pixels.
        * `path`: The relative location of the image file within the `images/small/` directories. This path is composed of lowercase hexadecimal characters (`0-9a-f`), which also uniquely identify the images. The first two characters of the `image_id` are used to construct a hierarchical file structure, optimizing directory management by reducing the number of images per directory. The majority of images are in `.jpg` format, with a few exceptions in `.png`.

* **`abo-images-small/images/small/`**:
    * The directory containing the actual image files. As mentioned, image paths follow a two-character hexadecimal hierarchy for efficient file system navigation.

### `abo-listings.tar` (83 MB)

This archive comprises product listings and their rich metadata, providing textual context for the images.

* **`abo-listings/listings/metadata/`**:
    * Contains 16 JSON files (from `listing_1.json` to `listing_f.json`).
    * Each JSON file contains multiple JSON objects, not initially wrapped in a list. Each object represents a product listing and can feature up to 28 fields describing the product.
    * **Key Fields for VQA Generation**: While many fields are present, our project specifically focused on a subset crucial for generating high-quality VQA pairs:
        * `bullet_point`: Important features of the product (e.g., "Water-resistant, Lightweight").
        * `color`: Color of the product as text (e.g., "blue", "green").
        * `color_code`: HTML color code of the product's color (e.g., "#0000FF").
        * `fabric_type`: Description of the product's fabric (e.g., "Cotton", "Polyester").
        * `finish_type`: Description of the product's finish (e.g., "Matte", "Glossy").
        * `item_shape`: Description of the product's shape (e.g., "rectangle", "square").
        * `material`: Description of the product's material (e.g., "Plastic", "Metal").
        * `pattern`: Product pattern (e.g., "Striped", "Floral").
        * `product_description`: Detailed product description, often in HTML format.
        * `product_type`: Product category (e.g., "CELLULAR_PHONE_CASE", "BACKPACK").
        * `style`: Style of the product (e.g., "Modern", "Vintage").
        * `main_image_id`: The primary product image, provided as an `image_id`.
        * `other_image_id`: Other available images for the product, provided as `image_id`s.

## Workflow: From Raw Data to Curated Dataset

The data curation process was meticulously designed to transform raw ABO data into a structured dataset suitable for VQA model training.

1.  **Initial Data Processing & Filtering**:
    * The first step involved parsing the 16 `.json` files within `abo-listings/listings/metadata/`. Each file contained multiple JSON objects, not encased in a list, necessitating initial parsing to wrap them into a standard JSON list for easier programmatic access.
    * Subsequently, we filtered out irrelevant fields from each JSON object, retaining only the 12 fields listed above (e.g., `bullet_point`, `color`, `material`, `product_type`, `main_image_id`, `other_image_id`). This reduced noise and focused on attributes critical for VQA question generation.

2.  **Image and Metadata Integration**:
    * To consolidate all image associations for a given product listing, we merged the `main_image_id` and `other_image_id` fields into a new `all_image_id` field. This new field contains a comprehensive list of all `image_id`s relevant to a specific product listing.
    * For each `image_id` in the `all_image_id` list, we queried the `abo-images-small/images/metadata/images.csv` file to retrieve its corresponding `path`. This `path` was then joined with the root path (`abo-images-small/images/small/`) to reconstruct the full image file location.

3.  **Product Type Categorization & Distribution**:
    * Based on the `product_type` field, the filtered JSON objects were organized into separate directories, one for each unique product category. This process revealed a total of 576 distinct product categories.
    * To facilitate parallel processing and efficient dataset generation, these 576 category folders were equally distributed among three team members, with each assigned 192 categories.

4.  **VQA Pair Generation with Gemini**:
    * From each JSON object within a product category folder, an `image_id` was selected from the `all_image_id` field.
    * The corresponding image was retrieved using the path obtained from `images.csv`.
    * This image, along with the relevant JSON object (excluding the `all_image_id` field), was then sent to the Gemini model via a carefully crafted prompt (detailed in "Prompt Engineering") to generate high-quality Q&A pairs.

## Product Type Distribution Analysis (All the 12 Parts are Provided Here `/DataCuration/OriginalDataSumamry/Plots`)
![Part_1](https://github.com/user-attachments/assets/4248c1a8-0049-4a81-8ce6-dbd144cae5e7)
![Part_2](https://github.com/user-attachments/assets/9374ced6-229d-4569-a13e-b9424c27698a)
![part_11](https://github.com/user-attachments/assets/4792c32e-27e2-4222-b6de-cc88fab56705)
![part_12](https://github.com/user-attachments/assets/63e88a72-ac07-46f9-82ac-550111ca7afa)

Here we Have included the top 2 parts/groups and bottom 2 parts/groups,The remaining Parts/Group Graphs Can be accessed at The above mentioned Path
Upon analyzing the distribution of the 147,702 product listings with multilingual metadata and 398,212 unique catalog image across the 576 product types, a severe class imbalance was observed.

**Key Observations:**

* **Dominance of a Single Category**: The `CELLULAR_PHONE_CASE` category overwhelmingly dominates the dataset, accounting for approximately 64,853 entries, which is nearly 43% of the total dataset. This highlights a significant skew where a single product type represents a substantial portion of the data.
* **Extreme Long Tail**: In stark contrast, numerous product categories contain fewer than 5 listings each, collectively contributing a negligible 0.0033% to the entire dataset.
* **Implications**: Such extreme disparities in category representation can introduce significant bias during model training. A model trained on this imbalanced data would likely perform disproportionately well on overrepresented classes (like `CELLULAR_PHONE_CASE`) while underperforming on categories with limited samples. This compromises the model's generalization capabilities and equitable performance across a wide range of product types.

To ensure a more balanced and representative dataset, it was crucial to implement a robust sampling strategy to normalize the distribution of product types, thereby promoting diversity and preventing model bias.

## Sampling Strategy: Proportional Tiered Sampling Algorithm

The severe class imbalance, where a small subset of classes (e.g., `CELLULAR_PHONE_CASE` with ~150,000 unique images or 64,853 product listings) dominates, while the majority have only 1 to 2 samples, posed a significant challenge. To address this, we adopted a **Proportional Tiered Sampling Algorithm**. This approach selectively samples listings from each product type based on its frequency, aiming for a more balanced and representative training set without discarding valuable data from any category.

### Algorithm Overview

The algorithm proportionally samples product listings using a **logarithmic scaling function**. This design ensures:
* **Retention of Rare Classes**: Product types with very few listings are fully preserved to maintain diversity.
* **Smooth Downsampling of Common Classes**: Categories with moderate frequencies are downsampled based on their logarithmic scale, preventing moderate overrepresentation.
* **Capping of Dominant Classes**: Categories with extremely high frequencies are capped at a maximum number of samples to prevent excessive bias toward them.

### Sampling Formula

For each product type $c$ with $N_c$ instances, the number of samples to retain $S_c$ is computed using the following formula:

$$
S_c = \min\left(\max\left(k, \log_{10}(N_c) \cdot k \cdot \alpha\right), M\right)
$$

Where:
* $k$: Minimum number of samples to retain for any class (e.g. , 1). This ensures that even the rarest classes contribute to the dataset.
* $\alpha$: Scaling factor (e.g., 2.5) that adjusts the influence of the logarithmic scaling. A higher $\alpha$ means more samples are retained as $N_c$ increases.
* $M$: Maximum cap for the number of samples (e.g., 15,000). This prevents dominant classes from overwhelming the dataset, even after logarithmic scaling.
* $N_c$: Number of original samples in class $c$.
* $S_c$: Final number of samples to keep from class $c$.

### Key Properties

* **Rare classes** (where $N_c \leq k$) are fully preserved, ensuring that the model is exposed to a wide variety of product types, even those with limited examples.
* **Common classes** (with moderate frequencies) are downsampled based on the logarithmic scale, allowing for a controlled reduction in their representation while still acknowledging their higher frequency.
* **Dominant classes** (with high frequencies) are capped to prevent excessive bias, ensuring that the model does not overfit to these highly represented categories.

This strategy creates a more balanced and diverse dataset, crucial for training a robust VQA model that performs well across all product categories, not just the most prevalent ones.

## Prompt Engineering for VQA Generation

The prompt, located in `/DataCuration/MainCode/Prompt.txt`, was meticulously crafted to generate high-quality question-answer pairs. Its primary objective was to ensure a comprehensive exploration of both visual content and product metadata, aiming to produce diverse, non-redundant, and detailed questions.

**Key Considerations in Prompt Design:**

* **Holistic Understanding**: The prompt encouraged the Gemini model to integrate information from the image (visual attributes like shape, color, pattern) with the provided metadata (e.g., `bullet_point`, `material`, `product_description`).
* **Diversity of Question Types**: Questions were designed to cover various aspects, including:
    * **Object Recognition**: "What is the main object in the image?"
    * **Spatial Relationships**: "Where is the handle located on the bag?"
    * **Material Properties**: "What material is the product made of according to the description?"
    * **Dimensional and Quantitative Details**: Encouraging questions about specific dimensions or features (e.g., "What are the approximate dimensions of the product?" or "How many compartments does the bag have?").
    * **Functional and Stylistic Attributes**: "What is the style of this product?", "What are some key features mentioned in the bullet points?"
* **Contextual Accuracy**: The prompt guided the model to generate questions and answers that were directly supported by the image and the associated metadata, minimizing hallucinations.
* **Non-Redundancy**: Efforts were made to encourage the generation of unique questions for each image, avoiding repetitive inquiries.
* **Detail and Specificity**: The prompt aimed for questions that required a deeper level of understanding, moving beyond simple visual identification to infer properties or relationships.

This rigorous approach to prompt engineering guarantees that the generated dataset is well-rounded, rich in context, and capable of challenging VQA models, thereby enhancing their performance in real-world, multi-modal tasks with superior scope and precision.

## Model Selection: Llama vs. Gemini

A critical decision point was the choice between the Llama and Gemini models for generating our Q&A pairs. To make an informed choice, we conducted an experimental evaluation using 30,000 images for each model and manually assessed the quality of the generated Q&A pairs. We also used the BLIP model as a reference to calculate accuracy metrics.

### Llama Model

* **Pros**:
    * **Local Installation**: A significant advantage was its ability to be installed and run locally, eliminating the dependency on external API keys and allowing for continuous, 24/7 operation without downtime.
* **Cons**:
    * **Quality**: Despite its operational flexibility, we observed a noticeable drop in the quality of the generated Q&A pairs compared to the Gemini model. The questions and answers were less detailed, less contextually accurate, and less diverse.

### Gemini Model

* **Pros**:
    * **Superior Quality**: The Gemini model consistently demonstrated superior performance in generating high-quality, contextually accurate Q&A pairs.
    * **Advanced Capabilities**: It excelled not only in standard question generation but also showed remarkable capabilities in extracting precise measurements from images and generating relevant Q&A pairs based on these detailed attributes. This level of sophistication was a key differentiator.
* **Cons**:
    * **API Key Dependency & Rate Limits**: A significant limitation was its reliance on API keys, which imposed usage limits. Specifically, with a single API key, the model was capable of generating only approximately 15 Q&A pairs for every 750 images processed. This limitation arose because Gemini's high-quality output often required more detailed and resource-intensive analysis per image, leading to a capped number of pairs per API usage cycle, especially when dealing with complex questions like measurement extraction or contextual relationships.

### Decision and Rationale

After thoroughly evaluating both models, we ultimately chose **Gemini** over **Llama** for our Q&A task. Despite the challenges posed by API key limitations and generation rates, Gemini's unparalleled ability to produce high-quality, contextually accurate, and detailed Q&A pairs—including the precise extraction of measurements—provided a significant value proposition that outweighed the operational conveniences of Llama. Its advanced capabilities were deemed essential for achieving the project's goal of a top-tier VQA dataset.

To address Gemini's API limitations and efficiently scale the generation process while maintaining quality, we developed a novel API key management and error handling strategy, which is detailed in the subsequent section.

## Implementation Details: The Code (`/DataCuration/MainCode/Final.py`)

Our robust implementation was designed to manage API calls efficiently, handle errors gracefully, and ensure the continuous generation of Q&A pairs for a large dataset.

### API Key Management and Error Handling Strategy

* **API Key Pool**: We maintain an `api_key.txt` file that stores a pool of 8-10 API keys. An accompanying `.env` file is used to manage the currently active API key. The active key is always placed at the top of the list in `api_key.txt` and referenced by the `.env` file.
* **Error Resilience**: During execution, we encountered a variety of common API-related errors, including:
    * `quota exceeded`
    * `connection reset`
    * `service unavailable`
    * `deadline exceeded`
    * `unavailable`
    * `504 Gateway Timeout`
    * `503 Service Unavailable`
    * `429 Too Many Requests`
    These errors typically occur after processing approximately 21 images, indicating a temporary server sleep or rate limit activation.
* **Retry Mechanism**: To counteract these issues, our system implements a sophisticated retry mechanism:
    1.  Upon encountering an error, the system pauses for 30 seconds.
    2.  It then attempts up to 5 retries for the current API call.
    3.  If all 5 retries fail, or if the 6th attempt (first attempt + 5 retries) is unsuccessful, the current API key is deemed exhausted or problematic.
    4.  In such cases, the current API key is moved to the bottom of the `api_key.txt` list, and the process seamlessly switches to the next available API key from the top of the list. This ensures continuous operation by rotating through the pool of keys.

### Output Storage

* The generated question-answer (QA) pairs are stored persistently in a separate directory, organized as individual `.json` files.
* Each `.json` file is named after its corresponding `image_id` 
* Each file contains two key fields:
    * `image_path`: The relative path to the image file.
    * `qa_pairs`: A list of dictionaries, each representing a Q&A pair generated for that image.

This robust implementation strategy allowed us to navigate the complexities of API rate limits and temporary service interruptions, ensuring the efficient and continuous generation of a large-scale VQA dataset.

## Results and Dataset Statistics (`/DataCuration/FinalGeneratedDataSummary/plots`)
![Part_1](https://github.com/user-attachments/assets/24de018e-a7ad-4b35-80d3-f2d75912822c)
![Part_2](https://github.com/user-attachments/assets/541e59a1-968d-499b-a0c1-14ae1e31a2b5)
![Part_11](https://github.com/user-attachments/assets/2e82e65e-7c51-4869-9b37-dbf73fc4d20e)
![Part_12](https://github.com/user-attachments/assets/01dd7a5d-01cc-4280-a381-f00584c353d6)
Here we Have included the top 2 parts/groups and bottom 2 parts/groups,The remaining Parts/Group Graphs Can be accessed at The above mentioned Path

We successfully generated question-and-answer (Q&A) pairs for **169659 images**, distributed across **559 categories**.
* **Category Reduction**: We observed a slight reduction in the total number of categories from the initial 576. This can be attributed to multiple `image_id`s potentially sharing the same `product_type` but not being sampled, or some categories being excluded by the sampling algorithm if their count was extremely low and they didn't meet the minimum threshold after processing.
* **Fairness in Distribution**: Crucially, the implemented sampling algorithm ensured that the number of images within each category was properly sampled according to our proportional tiered strategy, maintaining a fair and balanced representation across the diverse product types. The detailed summary of product types and their corresponding counts can be found in `/Data Curation/FinalGeneratedDataSummary/category_counts.txt`.

## Train-Test Split Strategy

The train-test split was a critical phase aimed at creating robust training and evaluation sets. The primary challenge was to ensure the test dataset maintained a high standard of quality and sufficient samples for meaningful evaluation, especially given the initial class imbalance.

**Threshold Determination:**

![image](https://github.com/user-attachments/assets/65a04115-ea6a-427b-a73b-f4cc0ac1bb4f)

Upon analyzing the image distribution (as visually represented in the provided plots as well as the image showing classes with there respective number of images in it for less than 110), we observed a clear pattern indicating that categories with fewer than a certain number of images would not provide statistically significant evaluation. Consequently, we established a threshold of **20 images**.

**Data Splitting Methodology:**

1.  **Training Set**:
    * Included all product types, regardless of their image count. This ensures that the training model is exposed to the full diversity of product categories, even the rare ones.

2.  **Test Set**:
    * Only product types with **more than 20 images** were considered eligible for inclusion in the test set. This criterion guarantees that each category represented in the test set has a minimum level of statistical significance for evaluation.
    * From these eligible product types, **only 20% of their images** were randomly selected to form the test set. This approach ensures a manageable test set size while maintaining representativeness.

**Dataset Statistics:**
Following this strategy, we successfully extracted:
* **34.1k JSON files** containing question-and-answer (Q&A) pairs for the **test set**.
* The remaining **135k JSON files** formed the **train set**.

The screenshots illustrating this split can be found in `/Data Curation/train-test-split`.

**Batching Strategy for Training Data:**
Given the substantial size of the training dataset (135k JSON files), including them all in a single directory was not feasible for efficient model training and data loading. Therefore, we implemented a structured batching process:

* **Even Distribution**: The process evenly distributes `.json` files from multiple category folders (within `Batches/` directory, which is where the output of the data generation was stored) into smaller, manageable batches.
* **Batch Size**: Each batch was capped at approximately **10,000 files**.
* **Round-Robin Approach**: To ensure each batch contained a balanced mix of files from all available categories and prevented class imbalance within individual batches, we employed a round-robin approach. The system first collected all `.json` files from each category folder, shuffled them to introduce randomness, and then iteratively built batches by taking one file at a time from each category in a rotating manner.
* **Directory Structure**: Once a batch reached approximately 10,000 files, it was moved into a newly created folder under `master_train/`. This process continued until all files were distributed, maintaining class diversity across all generated batches and preventing skewness in any single training batch.

![WhatsApp Image 2025-05-13 at 12 38 40](https://github.com/user-attachments/assets/e613c652-ba68-45e1-94b5-499cd6716575)

* We successfully divided the entire training set into 13 batches
This comprehensive train-test splitting and batching strategy ensures a robust evaluation framework and an efficiently loadable training corpus for developing VQA models.

## 📦 Final Dataset Overview

We’ve carefully curated two high-quality datasets to ensure robust model training and evaluation. Each dataset is hosted on Kaggle and publicly accessible.

---

### 🧠 Training Set – `Master-Train`

This dataset forms the backbone of our learning process. It contains well-labeled, diverse, and balanced data samples crucial for developing a high-performing model.

🔗 [Access Master-Train Dataset](https://www.kaggle.com/datasets/biradar1913/master-train)
🔗 [Access Master-Train(CSV) Dataset](https://www.kaggle.com/datasets/biradar1913/master-traincsv)

### 🧪 Test Set – `Master-Test`

Our test set is designed to rigorously evaluate the model's generalization capabilities. It mirrors real-world scenarios to validate the model’s effectiveness post-training.

🔗 [Access Master-Test Dataset](https://www.kaggle.com/datasets/biradar1913/master-test)
🔗 [Access Master-Test(CSV) Dataset](https://www.kaggle.com/datasets/biradar1913/master-testcsv)

---


# Model Fine-tuning with LoRA

To enable a general-purpose Visual Question Answering model to perform well on our target domain, a crucial step is fine-tuning it on relevant data. This process involves adjusting the model's parameters to better understand the specific visual and linguistic patterns of the new task. To ensure this fine-tuning was efficient and resource-friendly, we leveraged Low-Rank Adaptation (LoRA). LoRA is a state-of-the-art technique that selectively updates only a small portion of the model's weights, providing a highly effective way to adapt large models like BLIP For Question Answering without the overhead of full fine-tuning. This section outlines the methodology and implementation details of applying LoRA to our pre-trained model.

### Base Model and Processor
The foundation of the fine-tuning process was the Salesforce/blip-vqa-base model and its corresponding processor, loaded from Hugging Face Transformers.

### Model Choices: Rationale for selected models and any alternatives considered

The primary rationale for this choice is that BLIP is a state-of-the-art vision-language model specifically pre-trained on large datasets for joint image-text understanding and generation, including a VQA task. This provides a strong, purpose-built starting point for fine-tuning. While larger models like BLIP-2 variants (`blip-2.7b`, `blip-7b`) offer potentially higher performance due to their increased parameter counts, they also demand significantly more computational resources (GPU memory and processing power) and longer training times, which were key considerations for a course project with potentially limited hardware access. Alternatives like **CLIP**, while excellent for image-text contrastive learning and zero-shot classification, are primarily encoders and not directly designed for generating textual answers to questions. Implementing VQA with CLIP would require pairing it with a separate language model and developing a more complex architecture and fine-tuning setup for sequence generation, which was beyond the scope and complexity desired for this project. Therefore, `blip-vqa-base`, combined with the parameter-efficient LoRA fine-tuning technique, offered the optimal balance between performance, computational feasibility, and direct applicability to the VQA task.

### Data Preparation and Loading

Effective training of the Visual Question Answering (VQA) model requires structuring and preprocessing the image-question-answer data into a format compatible with the BLIP model architecture. For this purpose, a custom PyTorch `Dataset` class, named `VQADataset`, was developed.

The dataset is organized with image files stored separately from their corresponding question-answer annotations. Annotations are provided in JSON files. Each JSON file contains metadata for an image, including its relative path (`image_path`), and a list of ai-generated question-answer pairs (`qa_pairs`) related to that image.

### `VQADataset` Class

The `VQADataset` class serves as the crucial interface between the raw dataset files and the PyTorch training/evaluation pipeline. Its primary responsibility is to efficiently load, process, and format the image-question-answer triplets into tensors that the BLIP VQA model can directly consume. This custom implementation provides flexibility in handling our specific data structure and preprocessing requirements.

The class is designed to perform the following key functions:

*   **Scanning Specified JSON Directories:** The dataset constructor (`__init__`) takes a list of directories (`json_dirs`) as input. It then recursively traverses these directories using `os.walk` to find all files ending with the `.json` extension. This allows the dataset to be built from multiple distributed annotation files, potentially organized into different batches or subsets. Each discovered JSON file is expected to contain the VQA annotations for a single image.

*   **Processing JSON Data:** For every located JSON file, the class loads its content using `json.load`. It extracts the `image_path`, which typically contains a relative path to the corresponding image file, and the `qa_pairs` list, which holds multiple dictionaries, each representing a single question-answer pair (`question` and `answer` keys) for that image.

*   **Constructing Full Image Paths:** Dataset annotations often use relative paths. The `VQADataset` class takes a `base_img_path` argument to the constructor. For each image entry in the JSON, it constructs the absolute path to the image file by joining the `base_img_path` and the relative `image_path`. A specific cleaning step is included to handle potential inconsistencies in the relative path string (e.g., removing a fixed prefix like `abo-images-small/` if present), ensuring that the constructed path correctly points to the image location within the dataset's image storage.

*   **Storing Image-Question-Answer Samples:** The core data structure within the dataset object is a list named `self.samples`. Each element in this list is a dictionary representing a single VQA sample. This dictionary contains the fully constructed `image_path`, the `question` string, and the `answer` string for one specific question-answer pair associated with that image. By flattening the JSON structure into individual question-answer-image samples, we create a flexible dataset where each item directly corresponds to one prediction task for the model.

*   **Preprocessing in `__getitem__`:** The `__getitem__` method is invoked by the PyTorch `DataLoader` to retrieve and preprocess a single sample at a given index.
    *   **Image Loading and Formatting:** The image file specified by the sample's `image_path` is opened using the Pillow library. It is immediately converted to the `RGB` format (`.convert("RGB")`). This ensures that all images have a consistent three-channel structure, which is standard input for most vision models, regardless of their original format (e.g., grayscale).
    *   **Multimodal Input Processing:** The BLIP processor (`processor` object, initialized outside the class) is the primary tool for transforming the raw image and question into numerical tensors suitable for the BLIP model. The processor handles both image transformations (such as resizing and normalization based on the pre-trained model's requirements) and text tokenization (converting the question string into a sequence of numerical token IDs). The `return_tensors="pt"` argument ensures the output is PyTorch tensors. Padding (`padding="max_length"`) and truncation (`truncation=True`) are applied to the tokenized question to ensure all input sequences have a uniform length of `max_length=128`. This is essential for efficient batch processing by the model.
    *   **Answer Tokenization (Labels):** The correct answer string for the VQA sample is tokenized separately using the BLIP processor's tokenizer. Similar to the question, padding and truncation are applied to ensure the answer sequence has a fixed length of `max_length=10`. This tokenized answer sequence serves as the ground truth `labels` tensor that the model will be trained to generate.
    *   **Tensor Formatting:** The output tensors from the processor, which initially have a batch dimension of size 1 (due to `return_tensors="pt"` processing a single sample), are squeezed using `.squeeze(0)` to remove this dimension. This is the expected format for individual samples within a `Dataset` when used with a `DataLoader` for batching. The `inputs_embeds` tensor, if generated by the processor, is explicitly removed as the BLIP model variant used here takes token IDs (`input_ids`) as input for the language part, not embeddings.

By implementing these steps, the `VQADataset` class effectively bridges the gap between our raw data and the BLIP model, making the fine-tuning process straightforward within the PyTorch and Hugging Face Transformers ecosystems.




###  LoRA Configuration
 We leveraged the `peft` library from Hugging Face, which provides a straightforward way to apply LoRA to models from the Transformers library.

The specific LoRA configuration used for fine-tuning was carefully chosen to balance efficiency and performance:

*   **`r` = 8:** This parameter defines the **rank** of the low-rank update matrices (denoted as A and B). LoRA approximates the weight update matrix of a layer ($\Delta W$) by the product of two much smaller matrices, $A$ and $B$, where $\Delta W \approx BA$. The size of these matrices are $d \times r$ and $r \times k$ respectively, where $d \times k$ is the size of the original weight matrix $W$, and $r$ is the rank. A higher rank allows for a more expressive update but increases the number of trainable parameters ($r \times (d+k)$). A rank of 8 is a common starting point that generally provides a good balance between parameter efficiency and the ability to capture necessary adaptations.

*   **`lora_alpha` = 32:** This is the **scaling factor** for the LoRA updates. The scaled update added to the original weight matrix is $(BA) \cdot (lora-alpha / r)$. The `lora_alpha` parameter, in conjunction with `r`, determines the magnitude of the LoRA updates applied during fine-tuning. Using `lora_alpha = 32` with `r = 8` means the updates are scaled by a factor of $32 / 8 = 4$. This scaling helps to prevent the LoRA updates from being too small, especially when using a lower rank.
  
*   **`target_modules` = \[`'qkv'`, `'projection'`]:** This specifies which modules (layers) within the pre-trained BLIP model the LoRA matrices are injected into. `'qkv'` typically refers to the linear layers responsible for projecting the input embeddings into Query, Key, and Value vectors in the self-attention mechanism. `'projection'` likely refers to the output projection layer within the attention mechanism. Applying LoRA to these attention-related layers is a common and effective strategy because the attention mechanism is critical for the model's ability to understand relationships within the input (image features and text tokens) and generate contextually relevant outputs. Focusing fine-tuning on these layers allows the model to adapt its core understanding and generation capabilities to the new data.

*   **`lora_dropout` = 0.05:** This sets the **dropout probability** applied to the outputs of the LoRA update matrices ($BA$) during training. Dropout is a regularization technique where, during each training step, a fraction of the outputs from a layer are randomly set to zero. Applying dropout to the LoRA updates helps to prevent overfitting by making the model less reliant on any single LoRA parameter, encouraging it to learn more robust representations. A value of 0.05 represents a small amount of dropout, indicating a light form of regularization.

*   **`bias` = `'none'`:** This parameter controls whether bias terms in the target modules are also fine-tuned. Setting `bias` to `'none'` means that only the weight matrices within the specified `target_modules` are modified via the LoRA updates ($BA$), while the bias vectors of those layers are kept frozen. This is a standard practice in LoRA as fine-tuning only the weights often yields sufficient performance improvements, further contributing to parameter efficiency.

This specific LoRA configuration was chosen to efficiently adapt the BLIP VQA model, focusing the trainable parameters on key attention mechanisms with controlled scaling and mild regularization, thereby facilitating effective fine-tuning within the available computational constraints.

### Iterative Training Strategy

Given the substantial size of the complete VQA dataset, fine-tuning the BLIP model in a single pass over all data presented significant computational and memory challenges. To address this, a multi-stage, **iterative training strategy** was implemented. This approach involved dividing the dataset into smaller, manageable chunks and training the model sequentially on these batches, building upon the learning from previous stages.

The overall dataset was logically segmented into **14 distinct 'master' batches**. The fine-tuning process proceeded as follows:

1.  **Initialization:** Training commenced either from the base pre-trained `blip-vqa-base` model checkpoint (for the very first batch) or by loading the weights and configuration from the fine-tuned model checkpoint saved at the completion of the *previous* master batch's training. This allowed the model to continue learning and adapting from its most recent state.

2.  **Training on Master Batch N:** For each iteration, the model was fine-tuned exclusively on the data contained within the current 'Master Batch N'. The training was conducted using a standard deep learning loop: for each batch of data from the `VQADataset`, a forward pass was performed to obtain predictions, the loss (cross-entropy, as is typical for sequence generation tasks like VQA) was computed, gradients were calculated via backpropagation, and the model's trainable parameters were updated using the **AdamW optimizer** with a specified learning rate of **`lr` = 10e-5**.

3.  **Checkpointing and Model Versioning:** To ensure robustness against interruptions and to facilitate the iterative process, comprehensive checkpoints were saved regularly and versioned.

    *   Periodically, after a set number of training steps (e.g., every 1000 steps, as indicated by the code comment `save_every = 1000`), and definitively at the end of processing each master batch, the following were saved to a designated directory:
      
        *   The model's fine-tuned weights and configuration files using `model.save_pretrained(save_path)`.
        *   The processor's configuration and tokenizer files using `processor.save_pretrained(save_path)`.
          
        *   A dedicated 'training state' file (`training_state.pt`) using `torch.save`. This critical file included the state of the optimizer (`optimizer.state_dict()`), the current training step counter (`step_counter`), and the history of recorded losses (`loss_history`). Saving the optimizer state and step counter is essential for seamlessly resuming training exactly where it left off, preventing loss of progress.
          
    *   Each time a master batch's training was completed and saved, the output directory was named following a versioning convention: `/kaggle/working/model_latest_vN`. The 'vN' denotes the version number, indicating that the model within this directory has been trained up to and *including* the data from Master Batch 'N'.
      
    *   These versioned model checkpoints were then organized and stored within a **Kaggle dataset**. This centralized storage within a Kaggle dataset provided a convenient way to manage different iterations of the fine-tuned model, track progress across versions, and easily load a specific version as the starting point for subsequent training batches or for evaluation. The training state file (`training_state.pt`) was saved within the corresponding versioned model directory in the Kaggle working environment before being potentially included in the dataset.


5.  **Global Validation:** Upon the successful completion of training on each individual master batch, the fine-tuned model's performance was evaluated on a **global test dataset**. This test set consisted of approximately **482,036 samples** (as indicated by your evaluation cell output) and remained constant throughout the iterative process. Validating on a separate, large test set allowed for monitoring the model's generalization capabilities and tracking the cumulative impact of training on each successive data batch across the entire problem space, rather than just the performance on the currently processed batch.

6.  **Iteration Transition:** To move from training on 'Master Batch N' to 'Master Batch N+1', the `load_path` for the next training run was set to the `save_path` of the just-completed run. This ensured that the training for the next batch started from the weights and optimizer state of the model that had finished training on the previous batch. Concurrently, the `json_root_dir` in the dataset loading configuration was updated to point to the directory containing the data for 'Master Batch N+1'.

This sequential training process on different data batches allowed the model to incrementally process and learn from the entire large dataset, effectively managing memory and computational resources. Furthermore, the periodic and end-of-batch checkpointing provided resilience and enabled continuous performance tracking via the global validation step, offering insights into how the model improved as it was exposed to more data over successive batches.


### Performance Trends Across Iterations

To monitor the effectiveness of the iterative fine-tuning process and understand how the model's performance evolved as it was exposed to data from each successive master batch, we tracked key evaluation metrics on the constant global test dataset after training was completed for each batch. This provides a clear visualization of the learning progress.

The following graph illustrates the trend of two primary evaluation metrics: **Exact Match (EM)** and **BERTScore F1**, measured on the global test set after the model was fine-tuned up to and including the data from each respective master batch (from Batch 1 up to Batch 14). The performance of the base `blip-vqa-base` model *before* any fine-tuning serves as the initial baseline.

![image](https://github.com/user-attachments/assets/34cbbe57-6e57-45cb-af3c-c094e0386f27)
(See Figure: Performance Metrics Across Training Batches)

As anticipated with successful fine-tuning, the graph shows a general **increasing trend** for both the Exact Match (EM) and BERTScore F1 metrics as the model is trained on more data across subsequent batches.
*   **Exact Match (EM)** directly reflects the model's ability to produce answers that precisely match the ground truth. An increasing EM indicates that the model is becoming more precise in its factual recall and language generation for the specific VQA task.
*   **BERTScore F1** measures the semantic similarity between the generated and ground truth answers, providing a more nuanced assessment of answer quality. A rising BERTScore F1 suggests that even when the generated answer isn't an exact match, its meaning is becoming progressively closer to the correct answer's meaning.

This upward trend validates our iterative training strategy and demonstrates that training on each additional batch of data contributed positively to the model's overall performance and generalization capability on the unseen test set. The initial performance (Batch 0 or Base Model) serves as a valuable reference point to highlight the improvement gained through the fine-tuning process.




##  Evaluation

After the full iterative training cycle (or at intermediate stages after each master batch), the model was evaluated on a dedicated test dataset.

### Evaluation Setup
- The fine-tuned model and processor were loaded from their saved path (e.g., /kaggle/working/model_latest_v8).
- Hugging Face Accelerator was used for potentially distributed inference.
- A VQADataset instance was created for the test data (e.g., from /kaggle/input/master-test/test_dataset).
- A DataLoader was used for batching the test data.
- Mixed-precision inference (autocast) was used for speed.

### How Evaluation Works

1.  **Initialization:**
    * The model is set to evaluation mode (`model.eval()`).
    * Gradient calculations are disabled (`torch.no_grad()`) for efficiency.
    * Automatic Mixed Precision (`autocast`) is enabled if supported, reducing memory usage and potentially speeding up inference.
    * Crucially, evaluation state (processed batch indices, collected predictions, correct/total counts) is loaded from a `resume_file` if it exists. Otherwise, evaluation starts fresh. Variables like `initial_batches_processed`, `processed_indices`, `predicted_all`, `true_all`, `correct`, and `total` are initialized or populated based on the loaded state.

2.  **Iteration with Resume:**
    * The code iterates through the batches provided by the `test_loader`.
    * A `tqdm` progress bar is used to display progress, initialized to show the number of batches already processed (`initial_batches_processed`) from the resume state.
    * For each `batch_idx`, a check is performed: `if batch_idx < initial_batches_processed:`. If true, this batch was processed in a previous run.
    * If a batch is identified as already processed, `pbar.update()` is called to manually advance the progress bar, and `continue` skips the processing logic for this batch.

3.  **Batch Processing (for unprocessed batches):**
    * Input data (`pixel_values`, `input_ids`, `attention_mask`) is moved to the appropriate `device`.
    * The model's `generate` method is called with the input data to produce `generated_ids`. Generation parameters (`max_length`, `do_sample=False`, `num_beams=1`) control the output.
    * The `generated_ids` are decoded into human-readable strings (`predicted_answers`) using the `processor`.
    * Ground truth `answers` (labels) are also decoded into strings (`decoded_answers`).

4.  **Accuracy Calculation and Storage:**
    * The code iterates through each sample within the current batch, comparing the predicted and true answers.
    * Answers are cleaned by stripping whitespace and converting to lowercase (`.strip().lower()`) for case-insensitive comparison.
    * The cleaned predicted and true answers are appended to `predicted_all` and `true_all` lists, respectively.
    * If the cleaned predicted answer exactly matches the cleaned true answer, the `correct` counter is incremented.
    * The `total` counter (representing total samples processed so far) is incremented.

5.  **State Saving (Checkpointing):**
    * The current `batch_idx` is added to the `processed_indices` set.
    * The evaluation state is periodically saved to the specified `resume_file`. The saving condition triggers roughly every 1000 total samples processed (`total % 1000 < batch_size and total >= 1000`) and explicitly at the very end of the evaluation loop (`batch_idx == batch_count - 1`).
    * The saved state includes the list of processed batch indices (`indices`), the accumulated predictions (`predicted`), true answers (`true`), and the current correct (`correct`) and total (`total`) counts, stored as a JSON object.






## Evaluation Metrics

Evaluating Visual Question Answering models requires a robust set of metrics that can capture the nuances of generating natural language answers based on visual input. The following metrics were used to provide a multifaceted analysis of the models' capabilities:

### Standard Metrics

* **Exact Match (EM)**
    * **Indication:** Measures the percentage of generated answers that are an exact, character-for-character match to one of the ground truth reference answers.
    * **Justification:** Essential for evaluating questions with definitive, short answers where precision is paramount. It provides a clear measure of the model's ability to produce factually correct and precisely phrased responses for specific queries.
    * **Limitation for one-word VQA:** While suitable for questions with a single, unambiguous one-word answer, it harshly penalizes minor variations (e.g., singular vs. plural if not specified, or an equally valid synonym not present in the ground truth). It offers no partial credit for semantically very close but not identical single-word answers.

### Additional Metrics

* **BERTScore - Precision**
    * **Indication:** Quantifies how much of the generated answer is semantically similar to the reference answers, leveraging contextual embeddings from BERT. A higher score indicates that the model's output is relevant and avoids generating irrelevant information.
    * **Justification:** Moves beyond simple word overlap to assess semantic equivalence. Crucial for VQA as valid answers can be phrased in multiple ways. It helps to penalize the inclusion of incorrect or unsubstantiated details in the generated response.
    * **Limitation for one-word VQA:** For a single generated word, precision will largely indicate if that word is semantically aligned with the reference word(s). However, the complexity of BERT embeddings might be excessive for single-word comparisons and could potentially assign high precision to a semantically related but factually incorrect single word.

* **BERTScore - Recall**
    * **Indication:** Measures the extent to which the generated answer covers the information present in the reference answers, using BERT embeddings for semantic comparison. A higher score suggests the model is providing comprehensive answers that capture the key aspects of the ground truth.
    * **Justification:** Important for evaluating responses to open-ended questions that may require describing multiple elements or aspects of the image. It assesses if the model is effectively extracting and presenting the relevant information from the visual context.
    * **Limitation for one-word VQA:** If both generated and reference answers are single words, recall behaves very similarly to precision. If the model is expected to produce only one word, its ability to "cover" more information (which recall measures) is inherently limited, making this aspect less informative than for longer answers.

* **BERTScore - F1**
    * **Indication:** The harmonic mean of BERTScore Precision and Recall, offering a balanced measure of semantic similarity between the generated and reference answers.
    * **Justification:** Provides a single, robust score that reflects the overall semantic overlap and relevance of the generated answer, accounting for both the precision of the generated content and the coverage of the reference information. Often considered a primary metric for semantic evaluation.
    * **Limitation for one-word VQA:** Inherits limitations from BERTScore Precision and Recall for single words. While it balances semantic presence and relevance, it might still score a single, semantically similar but incorrect word highly. Its nuanced balancing act is more impactful for multi-word answers.

* **BARTScore**
    * **Indication:** Utilizes a pre-trained BART model to evaluate the quality of the generated text by assessing its likelihood within the BART language model, potentially capturing aspects like fluency, coherence, and factual consistency in a generation-aware manner.
    * **Justification:** Offers a more advanced, model-based evaluation that goes beyond surface-level text matching. It can provide insights into the naturalness and quality of the generated language, which is important for user-facing VQA applications.
    * **Limitation for one-word VQA:** Concepts like fluency and coherence are minimally applicable to single-word answers. BARTScore might favor common words over rare but correct single-word answers due to the language model's training distribution. Its strengths in evaluating the structure of generated text are underutilized for single tokens.

* **BLEU Score**
    * **Indication:** Measures the n-gram overlap between the generated answer and the reference answers, with a penalty for overly short generations. Primarily assesses the precision of word sequences.
    * **Justification:** A traditional metric for evaluating text generation quality, particularly useful for assessing how well the model replicates common phrases and word combinations found in human-provided answers. While sensitive to exact phrasing, it offers a foundational measure of textual similarity.
    * **Limitation for one-word VQA:** For single-word answers, BLEU (especially BLEU-1) essentially reduces to an exact match. Higher-order n-grams (BLEU-2, BLEU-3, BLEU-4) will typically be zero if the generated and reference answers are single words (unless they are identical), offering little discriminative power. The brevity penalty is also less relevant.

* **ROUGE-L**
    * **Indication:** Focuses on the Longest Common Subsequence (LCS) between the generated and reference answers, measuring the overlap in the longest shared sequence of words, irrespective of their order. Provides an F1-like score based on LCS.
    * **Justification:** Relevant for evaluating answers where the order of words might vary but the presence of a significant common sequence indicates shared content. Useful for assessing if the model captures the main informational flow or key phrases from the reference.
    * **Limitation for one-word VQA:** If both the generated and reference answers are single words, ROUGE-L effectively becomes a binary exact match (score 1 if identical, 0 otherwise). It cannot capture semantic similarity between different single words.

* **METEOR**
    * **Indication:** Calculates an alignment-based score considering exact word, stem, synonym, and paraphrase matches between the generated and reference answers. Designed to correlate better with human judgments than just n-gram overlap.
    * **Justification:** Addresses the limitations of purely surface-level metrics by incorporating semantic equivalence through synonymy and paraphrasing. Provides a more human-like evaluation of answer correctness when there are variations in wording.
    * **Limitation for one-word VQA:** While its ability to match synonyms is beneficial for single-word answers (making it more flexible than EM), the more complex aspects of METEOR like fragmentation and alignment penalties are less impactful for single words. It might give a good score to a synonym that is technically correct but not the most appropriate or common answer.

* **Jaccard Similarity**
    * **Indication:** Measures the ratio of the intersection to the union of the sets of unique tokens in the generated and reference answers. Indicates the degree of overlap in the vocabulary used.
    * **Justification:** Offers a simple, set-based measure of token overlap. Useful for understanding the common words shared between the generated and ground truth answers, providing a basic indication of content overlap ignoring word order and frequency.
    * **Limitation for one-word VQA:** For single-word answers, Jaccard Similarity becomes binary: it is 1 if the single generated word is identical to the single reference word, and 0 if they are different. It cannot distinguish between a completely unrelated word and a semantically close synonym.

* **Sørensen–Dice Coefficient**
    * **Indication:** Another set-based metric ($2 \times |A \cap B| / (|A| + |B|)$) quantifying the overlap between the sets of tokens in the generated and reference answers. Similar to Jaccard Similarity but can be less sensitive to the size of the sets.
    * **Justification:** Provides an alternative measure of token overlap, reinforcing the analysis of shared vocabulary between the model's output and the reference answers.
    * **Limitation for one-word VQA:** Similar to Jaccard Similarity, for single-word answers, this coefficient becomes 1 if the words are identical and 0 if they are different. It does not provide any partial credit for semantic similarity for non-identical single words.

* **LCS Ratio**
    * **Indication:** The ratio of the length of the Longest Common Subsequence (LCS) to the length of the reference answer. Indicates the proportion of the reference answer's word sequence captured by the generated answer.
    * **Justification:** A straightforward metric focusing specifically on the extent to which the generated answer preserves the order and content of the longest common sequence of words from the reference, offering insight into sequential overlap.
    * **Limitation for one-word VQA:** If the reference answer is a single word, the LCS ratio will be 1 for an exact match and 0 for any non-match. It offers no nuance for single words that might be semantically related but not identical.

* **Fuzzy Matching Score**
    * **Indication:** Measures the similarity between strings that may contain minor differences, such as typos or slight variations in spelling. Scores are based on the number of edits required to match the strings.
    * **Justification:** Important for VQA evaluation to account for potential minor errors in the model's output that do not fundamentally alter the correctness or meaning of the answer. Prevents penalizing models for small textual imperfections.
    * **Limitation for one-word VQA:** While useful for typos, it might incorrectly assign high similarity to single words that are orthographically close but semantically distinct (e.g., "horse" vs. "house"). For single-word answers, it's crucial that the "fuzziness" doesn't obscure actual incorrectness beyond minor misspellings of the correct word.

* **VQA Accuracy**
    * **Indication:** A standard VQA-specific metric that considers a generated answer correct if at least 3 out of 10 human annotators provided that answer as a ground truth.
    * **Justification:** Developed to handle the inherent subjectivity and variability in VQA answers. It provides a more realistic assessment of performance by acknowledging that multiple valid answers can exist for a single image-question pair.
    * **Limitation for one-word VQA:** If the pool of human-annotated answers for a given question primarily consists of specific single words, this metric might not fully credit an equally valid, synonymous single-word answer if that synonym wasn't provided by at least three annotators. Its effectiveness for unique one-word answers depends heavily on the diversity and exhaustiveness of the collected ground truth answers.


### Proposed Metrics

* **Visual-Contextual Consistency Score (VCCS)**
    * **Indication:** A metric designed to evaluate how well the generated answer aligns not just with the text of the question and reference answers, but also with the visual content of the image. It aims to assess if the entities, attributes, and relationships mentioned in the answer are truly present and consistent with the image.
    * **Justification:** Crucial for VQA, as it directly measures the model's ability to ground its linguistic response in the visual information. High VCCS indicates that the model is genuinely "seeing" and reasoning about the image to produce its answer, rather than just relying on language priors. This is vital for building reliable and trustworthy VQA systems.

* **Token-Level Overlap**
    * **Indication:** Provides a fine-grained analysis of the overlap between the generated and reference answers at the individual token level. This could involve categorizing overlapping tokens (e.g., by part of speech) or analyzing their distribution.
    * **Justification:** Offers a diagnostic tool to understand *what kind* of information the model is successfully transferring or generating compared to the ground truth. Helps in identifying specific areas of strength or weakness in the model's linguistic output related to the visual input.

By employing this comprehensive set of metrics, we aim to provide a thorough and insightful evaluation of the base and fine-tuned BLIP models, highlighting their performance across various dimensions of VQA.

## Baseline Evaluation Results

The evaluation of the base BLIP model yielded the following results across the 15 metrics:

| Metric                            | Score     |
| :-------------------------------- | :-------- |
| ✅ Exact Match (EM)                 | 13.28%    |
| 🤖 BERTScore - Precision          | 0.9487    |
| 🤖 BERTScore - Recall             | 0.9310    |
| 🤖 BERTScore - F1                 | 0.9388    |
| 🔹 BLEU Score                     | 0.0248    |
| 🔹 ROUGE-L                        | 0.1418    |
| 🔹 METEOR                         | 0.0853    |
| 🔹 Jaccard Similarity             | 0.1369    |
| 🔹 Sørensen–Dice Coefficient      | 0.1387    |
| 🔹 LCS Ratio                      | 0.1370    |
| 🔹 Token-Level Overlap            | 0.1369    |
| 🔹 Fuzzy Matching Score           | 0.2818    |
| 🔹 VQA Accuracy                   | 13.28%    |
| 🔹 Visual-Contextual Consistency Score (VCCS) | 0.1373    |
| 🔹 BARTScore                      | -6.3153   |

These results serve as a baseline to understand the performance of the foundational model before fine-tuning. Subsequent evaluations of the fine-tuned versions will be compared against these scores.

## Final Fine-tuned Model Evaluation Results (Version 13)

The final fine-tuned BLIP model (Version 13) demonstrated improved performance across the majority of the evaluation metrics compared to the baseline model. The results are as follows:

| Metric                            | Score     |
| :-------------------------------- | :-------- |
| ✅ Exact Match (EM)                 | 20.53%    |
| 🤖 BERTScore - Precision          | 0.9564    |
| 🤖 BERTScore - Recall             | 0.9383    |
| 🤖 BERTScore - F1                 | 0.9464    |
| 🔹 BLEU Score                     | 0.0374    |
| 🔹 ROUGE-L                        | 0.2143    |
| 🔹 METEOR                         | 0.1105    |
| 🔹 Jaccard Similarity             | 0.2091    |
| 🔹 Sørensen–Dice Coefficient      | 0.2109    |
| 🔹 LCS Ratio                      | 0.2093    |
| 🔹 Token-Level Overlap            | 0.2091    |
| 🔹 Fuzzy Matching Score           | 0.3840    |
| 🔹 VQA Accuracy                   | 20.53%    |
| 🔹 Visual-Contextual Consistency Score (VCCS) | 0.2094    |
| 🔹 BARTScore                      | -5.8659   |

These results indicate the effectiveness of the fine-tuning process in improving the model's ability to generate accurate and relevant answers for the VQA task, as captured by a diverse set of evaluation metrics.

# BLIP Model Fine-tuning for Visual Question Answering (VQA) Performance Analysis

## Detailed Results

![1](https://github.com/user-attachments/assets/830f7142-0c6d-46b0-90b5-8493553f0582)


Performance comparision of our 2 Best Models(V-7 and V-13)


![44](https://github.com/user-attachments/assets/643c4656-da01-4923-a60a-0d9e0777c7e5)  


![2](https://github.com/user-attachments/assets/476c9c23-934a-4ebc-b70a-3bef9c662b3a)


![22](https://github.com/user-attachments/assets/d5817db2-b884-404b-a5a6-f7b0654f15ed)



The detailed summary of evaluation metrics for each of the 13 versions and the baseline can be found in the directory: `/EvaluationMetrics/Results`.

This Part summarizes the performance analysis of a BLIP model fine-tuned for Visual Question Answering (VQA) across 13 iterative versions. The fine-tuning process aimed to enhance the model's ability to understand visual content and generate accurate text-based answers to questions about images.
The analysis reveals that the fine-tuning process resulted in a significant overall improvement in the BLIP model's VQA performance compared to the baseline. A clear trend of initial rapid gains, followed by fluctuations, a peak in performance, a subsequent dip, and a strong final recovery is observed across most key evaluation metrics.

## Overall Trend

The fine-tuning process effectively boosted the model's ability to answer visual questions. The performance, measured across various metrics, increased sharply from the baseline, reached a peak around version 7, experienced a temporary decline in versions 11 and 12, and then achieved near-peak performance in the final version (v13).

## Key Metrics & Their Trends

Performance was evaluated using a suite of metrics, providing a comprehensive view of the model's capabilities:

* **VQA Accuracy / Exact Match (EM):**
    * The most critical metric for VQA, showing a dramatic increase from **Baseline (13.28%)** to **v1 (18.42%)**.
    * Accuracy climbed steadily, reaching its highest point at **v7 (20.72%)**.
    * Performance then fluctuated (v8-v10), dipped significantly in **v11 (19.22%)**, and hit a low at **v12 (17.99%)**.
    * The final **v13 (20.53%)** showed a strong recovery, closely approaching the v7 peak.

* **Text Generation Metrics (BERTScore F1, ROUGE-L, METEOR, BLEU, Jaccard Similarity, Sørensen–Dice Coefficient, LCS Ratio, Token-Level Overlap, Fuzzy Matching):**
    * These metrics evaluate the quality and overlap of generated answers with ground truth using various methods (semantic, n-gram, sequence, fuzzy matching).
    * Collectively mirrored the VQA Accuracy/EM trend: significant increases from baseline to v1, improvement towards the **v7 peak** (often maxing out here), followed the v11/v12 dip, and showed strong recovery in **v13** to near-peak levels.
    * **BERTScore F1** remained relatively high throughout (0.9388 baseline to 0.9477 peak), indicating the model quickly learned to generate semantically relevant answers.
    * **Fuzzy Matching** showed slightly different peaks at **v3 (0.3812)** and **v13 (0.3840)**, suggesting improved handling of minor variations in these versions.

* **Visual-Contextual Consistency Score (VCCS):**
    * A VQA-specific metric assessing answer-visual context alignment.
    * Followed a trend similar to VQA Accuracy/EM: improving from **Baseline (0.1373)**, peaking at **v7 (0.2117)**, dipping at **v12 (0.1836)**, and recovering strongly at **v13 (0.2094)**.
    * Indicates improved ability to generate answers consistent with the image content.

* **BARTScore:**
    * Evaluates generation quality using a BART model (less negative is better).
    * Improved from **Baseline (-6.3153)** to the **v7 peak (-5.8530)**.
    * Followed the general trend of fluctuation and the dip in v11/v12 (becoming more negative).
    * Showed a notable improvement in **v13 (-5.8659)**, returning to a value close to the peak.

## Notable Observations

* The initial fine-tuning step (Baseline to v1) provided the most significant single jump in performance across almost all metrics.
* **Version 7** represents the overall best-performing model in this sequence, achieving the highest VQA Accuracy and peak values for many other generation metrics.
* **Versions 11 and 12** show a clear regression in performance compared to earlier versions.
* The final **v13** model successfully recovered from the v11/v12 dip, achieving performance very close to the peak (v7) and representing a substantial improvement over the baseline.

The fine-tuning process successfully enhanced the BLIP model's VQA capabilities. The analysis across 13 versions demonstrated a clear positive trajectory, despite temporary dips. The final version (v13) achieved strong performance across all key metrics, significantly surpassing the original baseline and demonstrating the effectiveness of the fine-tuning dataset and process. Key performance indicators like VQA Accuracy, BERTScore F1, METEOR, ROUGE-L, and VCCS all reflect this overall positive trend with specific periods of rapid improvement, peak performance, temporary decline, and final recovery.

## 🧑‍💻 Team Members

| Name              | Student ID    
| :---------------- | :------------ 
| Aryaman Pathak    | IMT2022513 
| Rutul Patel       | IMT2022021   
| Shreyas Biradar   | IMT2022529   

---

## 🙏 Acknowledgments

This project is undertaken as part of the **[Course Name/Code, e.g., "Multimodal Machine Learning (CS XXXX)"]** course at **[Your University Name]**. We extend our gratitude to the course instructors and teaching assistants for their guidance and the opportunity to work on this exciting project.

---

## 📄 License

This project is licensed under the **[Choose a License, e.g., MIT License]** - see the `LICENSE` file for details.

---
