# VentureAI

![VentureAI Logo](./VentureAI.png)

We introduce Venture AI, your automated travel agent. The goal of Venture AI is to address the complexity of modern travel planning by creating a single application that provides personalized travel recommendations tailored to each user’s preferences and budget. Today’s large language models are very versatile, but they lack the specific customization needed to deliver precise suggestions for unique user needs. By utilizing LLMs and integrating them with real-time APIs from various travel-related sites, Venture AI will act as a comprehensive travel planner that can cater to a wide range of travel preferences.

This project addresses a common problem with travel planning by using new LLM techniques to simplify and centralize the planning process. Without LLMs, users must manually navigate different websites to organize flights and local activities, often making travel planning inefficient and demotivating. LLMs offer an advantage in this space due to their ability to handle and respond to natural language queries. They can process real-time user feedback to refine suggestions in an adaptable and user-friendly way.

Through these capabilities, Venture AI aims to transform how travelers organize their adventures, making the process more efficient, personalized, and enjoyable.

## Cleaning:
* All instances of `&quot;` have been replaced with `\"` to since we wouldn't want the model to output special formatting like that.
* All information specific to images inside double brackets `{{}}` has been removed, but the captions for the images currently are kept because they still contain useful information.
* There are two different variations of `[[]]`. The ones that contain the substring `File:` within them are file links, and those have been removed. Otherwise, they are links to other pages in WikiVoyage. For those, we just removed the brackets and kept the text.

**Note: all path commands assume that you are in the root folder of this repository, to standardize the provided path commands.**
## Data
* All of the data that use/generate/clean is located in the `./data_generation` folder.
* `./data_generation/data_compilation` contains our most recent csv files that are directly used in training, and the cleaning files used to create them.
* `./data_generation/wikivoyage` contains all of our Wikivoyage data. This is the data that we used to generate the fine tuning datasets.


## Training
If you want to run the code yourself to test it or fine tune the model in your own way:
The code for all of the models themselves is located in the `./fine_tuning` folder.
### Environment
All of the dependencies are located in `./environment.yml`. Once you have conda activated and ready to use, follow these steps:
1. `cd ./` (go to the root folder of the repository) 
2. `conda env create -f environment.yml`
3. `conda activate juypter`
You can check to make sure all the dependencies match with `conda env list`.
### Fine-tuning
To start fine-tuning the model:
1. `cd ./fine_tuning`
2. `torchrun --nproc_per_node={# GPUS} {filename.py}` 
 * `filename.py` is the file/model that you want to fine-tune. For example, you could run `torchrun --nproc_per_node=4 falcon7b.py` to fine-tune the Falcon7b model with 4 GPUs.
 * \# GPUS is the number of GPUs that you currently have on your system that you are going to use in parallel for training.