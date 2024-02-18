DISCOREV: crosstask knowledge DIStillation for COde REView
===============================

This is the replication package accompanying our paper, [**Improving the Learning of Code Review Successive Tasks with Cross-Task Knowledge Distillation**](https://arxiv.org/html/2402.02063v1), that is published in the *ACM International Conference on the Foundations of Software Engineering 2024 (FSE'24)*.

Overview
---
*DISCOREV* is a novel deep-learning architecture that employs cross-task knowledge distillation to address code review tasks. In our approach, we utilize a cascade of models to enhance both comment generation and code refinement models. The fine-tuning of the comment generation model is guided by the code refinement model, while the fine-tuning of the code refinement model is guided by the quality estimation model. We implement this guidance using two strategies: a feedback-based learning objective and an embedding alignment objective

This project is designed to be standalone and utilizes *Docker* for easy replication of results, as well as to facilitate the model's usability, modification, and execution.




Project structure
---

- ### Dockerfile
    The Dockerfile provides a containerized environment for seamless replication of results and simplified usage and modification of the model.

- ### comment_generation
    This folder contains the code to train DISCOREV on the *comment generation* task.
    ```
    ├── config.py               # defines and parses the configuration (i.e., list of parameters) of the program
    ├── dataset.py              # reads and pre-processes the dataset 
    ├── evaluation.py           # defines the evaluation metrics
    ├── evaluate_model1.py      # evaluation script of model 1
    ├── evaluate_model2.py      # evaluation script of model 2
    ├── models.py               # defines the architecture of DISCOREV model
    ├── smooth_bleu.py          # defines BLEU metrics
    ├── utils.py                # utils for model tratining and data pre-preprocessing 
    ├── main.py                 # main script for training DISCOREV model 
    ```

- ### code_refinement
    This folder contains the code to train DISCOREV on the *code refinement* task. It has a similar structure to the *comment_generation* folder.

- ### data
    This folder should contain the data used by the models (i.e., training, test, and validation datasets). We use [this dataset](https://zenodo.org/records/6900648) for our experiments.




Training
---

To facilitate the training of DISCOREV, it can be done inside a Docker container. 
Follow the steps below to build the Docker image and train DISCOREV.

1. Clone the DISCOREV repository:
    ```
    git clone https://github.com/OussamaSghaier/DISCOREV.git
    ```

2. Navigate to the DISCOREV directory:
    ```
    cd DISCOREV
    ```

3. Build the Docker image:
    ```
    docker build -t discorev-image .
    ```

4. Create a docker container from the built image and connect to the running Docker container:
    ```
    docker run -it --rm --gpus all -v /tmp/DISCOREV/data/:/app/data --name discorev-container discorev-image

    ```

    The above command should be run once (the first time) to create a container from the built image. 
    */tmp/DISCOREV/data* is the absolute path 
    For subsequent attempts, we should use the following commands to start and connect to the created docker container:
    ```bash
    docker start discorev-container
    ```

    To stop the container:
    ```
    docker stop discorev-container
    ```

    DISCOREV is now set up to be trained within a Docker container for your data, and you can easily start, connect, and stop the container as needed.

5. Train DISCOREV: Execute the following command to train DISCOREV
    ```
    python main.py 
        --train_file ../data/ref-train.jsonl 
        --test_file ../data/ref-test.jsonl 
        --valid_file ../data/ref-valid.jsonl
        --model_name_or_path microsoft/codereviewer 
        --num_epochs 30
        --batch_size 32
        --output_dir ./output
        --save-steps 5000
        --eval-steps 5000
        --log-steps 500
        --seef 1234
        --max_source_length 512 \
        --max_target_length 128 \
        --learning_rate 3e-4 \
        --gradient_accumulation_steps 3 \
    ```


Citing
---
If you use this code or DISCOREV, please consider citing us.

    @article{sghaier2024improving,
        title={Improving the Learning of Code Review Successive Tasks with Cross-Task Knowledge Distillation},
        author={Sghaier, Oussama Ben and Sahraoui, Houari},
        journal={arXiv preprint arXiv:2402.02063},
        year={2024}
    }


Contact us
---
We value your feedback and are here to assist you with any questions you may have regarding our paper. 
Please feel free to reach out to us:

- Email: oussama.ben.sghaier@umontreal.ca
- Website: https://oussamasghaier.github.io/



