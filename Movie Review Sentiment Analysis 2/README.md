This project uses the '**rotten_tomatoes**' dataset, which consists of movie reviews, by directly uploading it from HuggingFace. This project aims to train a classification model of movie reviews, specifically fine-tuning a pre-trained Transformers model.

The dataset contains 10,662 movie reviews, evenly split between positive and negative sentiments.

Here is a concise view of different models tried and their respective F1 macro scores:

+--------------------------------------------------------+----------------------------+----------------------------+-----------------------------------------------------------------------+--------------------------------------------------------------------+
| [BERT]{.underline}                                     | [RoBERTa]{.underline}      | [ALBERT]{.underline}       | [DeBERTa-base]{.underline}                                            | [DeBERTa-v3-large]{.underline}                                     |
+========================================================+============================+============================+=======================================================================+====================================================================+
| [Trial 1]{.underline}: Tokenizer of '`max_length=128`' | [Trial 1]{.underline}**:** | [Trial 1]{.underline}**:** | [Trial 1]{.underline}**:** Hyperparameter search                      | [Trial 1:]{.underline}                                             |
|                                                        |                            |                            |                                                                       |                                                                    |
| *Macro F1*: 0.836                                      | *Macro F1:* 0.863          | *Macro F1:* 0.794          | *Macro F1:* 0.887                                                     | *Macro F1:* 0.919                                                  |
+--------------------------------------------------------+----------------------------+----------------------------+-----------------------------------------------------------------------+--------------------------------------------------------------------+
| [Trial 2:]{.underline} Tokenizer of '`max_length=256`' |                            |                            | [Trial 2]{.underline}**:** DeBERTa-base + Layer Freezing / Unfreezing | [Trial 2:]{.underline} DeBERTa-large + Layer Freezing / Unfreezing |
|                                                        |                            |                            |                                                                       |                                                                    |
| *Macro F1*: 0.853                                      |                            |                            | *Macro F1:* 0.335 (Freezing + Unfreezing)                             | *Macro F1:* 0.912                                                  |
|                                                        |                            |                            |                                                                       |                                                                    |
|                                                        |                            |                            | *Macro F1:* 0.889 (Unfreezing only)                                   |                                                                    |
+--------------------------------------------------------+----------------------------+----------------------------+-----------------------------------------------------------------------+--------------------------------------------------------------------+
| [Trial 3]{.underline}: Tokenizer with dynamic padding: |                            |                            | [Trial 3]{.underline}**:** Adding optimizer                           |                                                                    |
|                                                        |                            |                            |                                                                       |                                                                    |
| *Macro F1*: 0.847                                      |                            |                            | *Macro F1:* 0.896                                                     |                                                                    |
+--------------------------------------------------------+----------------------------+----------------------------+-----------------------------------------------------------------------+--------------------------------------------------------------------+

From the overview of the models above, we can see that ALBERT was the worst performing model, while DeBERTa-v3-large was the best performing one. The details of each model's trials and errors can be found below:

### [**Model 1**]{.underline}**: BERT**

The first model chosen is BERT, which can understand the context of a word by looking at its surroundings and allows it to have a deep understanding of the language structure. Initially, I chose the simplest format of the BERT model as a starting point.

-   *Pretraining*: AutoTokenizer from transformers to tokenize the data into a format that BERT can understand. I tried fixed-length padding (128 and then 256), which when increased allows the model to capture more context in longer reviews. Then I introduced dynamic padding to improve the computational efficiency, as it dynamically adjusts sequence lengths.

-   *Training*: I utilized the Hugging Face 'trainer' API to fine-tune BERT and choose different epochs, batch sizes, and warm-ups with try outs, and by considering the size of the dataset and the complexity of the task.

-   *Evaluation* : Overall, the model took very long to run. I had to change the runtime from CPU to T4 GPU in order to get the training to run faster. The models gave better results than I expected.

The model with the fixed-length tokenizer of 256 showed the best macro score of 0.852 but had way higher computational overhead. - The model with dynamic padding showed slightly lower results with a macro score of 0.847 but faster processing times as it adjusts padding based on the actual need of each batch.

Thereefore, for the models to come, I kept the tokenizer with dynamic padding.

### [**Model 2**]{.underline}**: RoBERTa**

I continued with RoBERTa with the expectation of higher efficiency and effectiveness given that it trains on more data and for longer periods.

-   *Pretraining*: I adapted the tokenization process for RoBERTa, still using dynamic padding.

-   *Training*: 'Trainer' API, I configured the training process with specific epochs, batch sizes, and warm-up steps. Dynamic padding was integrated using 'DataCollatorWithPadding' for training across variable-length reviews. For the generalization of the model, I added regularization techniques, specifically weight decay, and dropout.

-   *Evaluation*: RoBERTa outperformed the initial BERT model with a macro of 0.86, showing better capabilities in analyzing movie reviews. In addition, the overall processing was faster than for BERT.

I did not experiment further with RoBERTa as I expected to have a higher score from the first model.

### [**Model 3**]{.underline}**: ALBERT**

Next, I explored ALBERT, which promises higher efficiency with similar or better performance.

-   *Pretraining*: Adapted tokenization for ALBERT, once again using dynamic padding.

-   *Training*: Introduced more specific training arguments. Adjusted the batch size to 16, epochs to 4, learning rate to 3e-5, and incorporated a step-based evaluation and saving strategy to better monitor the model's progress.

-   *Evaluation*: Despite higher efficiency, results were lower compared to the other models, with an F1 score of 0.79. This could be due to ALBERT's streamlined architecture, which, while efficient, can require more tuning or a larger dataset.

ALBERT performed the worst so far, so I also decided to move on and try a better-performing model such as DeBERTa.

### [**Model 4**]{.underline}**: DeBERTa-base**

The next model I tried out was DeBERTa-base. It ran for longer than any of the other models, and I incorporated different methods such as hyperparameter search and adding freezing/unfreezing layers.

-   *Pretraining*: DeBERTa-base adapted tokenization.
-   *Training*: Initially played around with training arguments, then set to specific ones found from hyperparameter search.
-   *Evaluation:* I used hyperparameter search immediately to find the best values for the model, and I also incorporated different methods to improve it further. On the other hand, it used way more computational power in comparison to the previous models. In terms of F1 score, the highest one received was: 0.896.

#### 1. Hyperparameter Search:

For DeBERTa, I conducted a hyperparameter search using Optuna for hyperparameter optimization. The search was used to find the optimal learning rates, number of training epochs, batch size, and warm step ups. Using hyperparameter search to create a base model, I received a F1 score of 0.887.

#### 2. Freezing/Unfreezing:

I incorporated layer freezing and unfreezing for fine-tuning of the model and increase the complexity of it.

-   When adding both freezing and unfreezing layers, the performance on the validation set dropped significantly, with F1 scores of 0.335.

-   When removing the freezing layers and keeping **only** the unfreezing layers, the F1 score showed to be 0.889, slightly higher than the other trials.

### [**Model 5**]{.underline}**: DeBERTa-v3-large**

The final model I tried out was DeBERTa-v3-large, which trains on more parameters and is overall better at capturing complexity and semantic meaning. I was expecting that this model would perform the best out of all, which was correct.

-   *Pretraining*: Tokenization for DeBERTa-v3-large.
-   *Training*: Performed better with lower values of hyperparameters, such as for no. of epochs = 2 and batch sizes of 8.
-   *Evaluation:* F1 score of the first trial was 0.918, highest out of all previous models.

I also added layer freezing/unfreezing for this model, but F1 score slightly decreased to 0.912. Therefore, I removed it. This entails that the model is powerful and performs well on its own, and the use of these layers improve the complexity of it without any performance gains.

### *Final Model:* DeBERTa-v3-large

The final and best performing model chosen was DeBERTa-v3-large, which showed better ability to understand and process natural language data. The key parameters fine-tuned for optimal performance include:

1.  `number of train epochs = 2`.
2.  `train and evaluation batch size = 8`.
3.  `learning rate = 1e-5`.

I organized the entire code in terms of:

1.  [Imports]{.underline} - all the necessary libraries for the code to run.
2.  [Functions]{.underline}:
    1.  '**tokenizer_function**': prepares text for the model by tokenizing.
    2.  '**compute_metrics**' function: calculates performance metrics (accuracy, precision, recall, F1 score) from predictions).
3.  [Main script:]{.underline}
    1.  Set up device for training (GPU if available).
    2.  Load and tokenize the "rotten_tomatoes" dataset.
    3.  Initializes DeBERTa-v3-large model with specific training configurations.
    4.  Trains and evaluates model using Hugging Face's **'Trainer'** API.
4.  [Results Processing]{.underline} - Outputs model performance on the test dataset and saves predictions in a CSV for review.

#### Conclusion:

After a series of experiments, DeBERTa-v3-large proved to be the standout model for our sentiment analysis project, with an F1 macro score of 0.918 on the 'rotten_tomatoes' dataset. The main challenge I faced was the intense computational demand, a crucial consideration for anyone looking to replicate or build upon this work. This exploration highlights the balance and importance between selecting the right model and managing computational resources effectively, in addition to inspiring further advancements in efficient NLP model deployment.
