# 🚀 Multimodal Visual Question Answering with Amazon Berkeley Objects (ABO) Dataset

## Project Overview

> This project focuses on developing a **Multimodal Visual Question Answering (VQA)** system using the rich **Amazon Berkeley Objects (ABO)** dataset. Our objective is to create a multiple-choice VQA dataset from ABO, establish and evaluate robust baseline models, and then significantly improve their performance through fine-tuning using the **Low-Rank Adaptation (LoRA)** technique. The ultimate goal is to build a high-performing VQA model capable of answering questions about visual data, which will be rigorously evaluated on a hidden dataset and an undisclosed metric.

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

## Product Type Distribution Analysis
![image](https://github.com/user-attachments/assets/c48cb497-5846-4f75-b556-9ec9453ad4ec)
![image](https://github.com/user-attachments/assets/0b309650-8194-4f88-b59b-c0898a1b3572)
![image](https://github.com/user-attachments/assets/1484554e-c6c0-456e-9e8c-8b2dfe51ecf3)
![image](https://github.com/user-attachments/assets/e99fbe82-f351-46c7-acf0-3761de4f2b52)

The remaining Parts Graphs Can be accessed at /DataCuration/OriginalDataSummary/Plots
Upon analyzing the distribution of the 147,702 product listings across the 576 product types, a severe class imbalance was observed.

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
* $k$: Minimum number of samples to retain for any class (e.g., 5). This ensures that even the rarest classes contribute to the dataset.
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

The prompt, located in `/Data Curation/MainCode/Prompt.txt`, was meticulously crafted to generate high-quality question-answer pairs. Its primary objective was to ensure a comprehensive exploration of both visual content and product metadata, aiming to produce diverse, non-redundant, and detailed questions.

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

## Implementation Details: The Code (`/Data Curation/MainCode/Final.py`)

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
* Each `.json` file is named after its corresponding `image_id` (e.g., `image_id_xxxx.json`).
* Each file contains two key fields:
    * `image_path`: The relative path to the image file.
    * `qa_pairs`: A list of dictionaries, each representing a Q&A pair generated for that image.

This robust implementation strategy allowed us to navigate the complexities of API rate limits and temporary service interruptions, ensuring the efficient and continuous generation of a large-scale VQA dataset.

## Results and Dataset Statistics
![image](https://github.com/user-attachments/assets/e1a829e4-cf1e-4cde-8efd-02ecd067ae72)
We successfully generated question-and-answer (Q&A) pairs for **177,375 images**, distributed across **559 categories**.

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
* The remaining **143k JSON files** formed the **train set**.

The screenshots illustrating this split can be found in `/Data Curation/train-test-split`.

**Batching Strategy for Training Data:**
Given the substantial size of the training dataset (143k JSON files), including them all in a single directory was not feasible for efficient model training and data loading. Therefore, we implemented a structured batching process:

* **Even Distribution**: The process evenly distributes `.json` files from multiple category folders (within `Batches/` directory, which is where the output of the data generation was stored) into smaller, manageable batches.
* **Batch Size**: Each batch was capped at **10,000 files**.
* **Round-Robin Approach**: To ensure each batch contained a balanced mix of files from all available categories and prevented class imbalance within individual batches, we employed a round-robin approach. The system first collected all `.json` files from each category folder, shuffled them to introduce randomness, and then iteratively built batches by taking one file at a time from each category in a rotating manner.
* **Directory Structure**: Once a batch reached 10,000 files, it was moved into a newly created folder under `master_train/`. This process continued until all files were distributed, maintaining class diversity across all generated batches and preventing skewness in any single training batch.
![image](https://github.com/user-attachments/assets/3ecc303c-d928-4cd4-914b-971eb98fc5a7)
This comprehensive train-test splitting and batching strategy ensures a robust evaluation framework and an efficiently loadable training corpus for developing VQA models.



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
