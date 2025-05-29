# Homework #3 Report
**MSAI-337: Parameter Efficient Fine-Tuning**  
**Name:** Nicole Birova, Jack Hanke, Daniel Plotkin
**Due Date:** June 8, 2025  

---

## Task 1: Full Fine-Tuning Baseline 

### Approach Overview

To establish a full fine-tuning baseline, we trained a custom multiple-choice classifier using `BERT-base-uncased` on the OpenBookQA dataset. In accordance with the assignment constraints, we did **not use `BertForMultipleChoice` or any pre-built multiple-choice modules**.

Each question-answer instance was processed into four separate BERT input sequences:

```
[CLS] <fact> <question stem> <choice text> [SEP]
```

- Each of the 4 sequences per question was encoded independently.
- For each encoded sequence, we extracted the `[CLS]` token's final-layer embedding.
- A shared linear classification layer converted each `[CLS]` vector into a scalar logit.
- The resulting 4 logits per question were concatenated and passed through a softmax.
- Cross-entropy loss was computed against the correct choice label.

The model was trained end-to-end on CPU for **3 full epochs**, fine-tuning **all** of BERT’s parameters (~110M total).

---

### Code Snippet: Forward Pass

```python
def forward(self, batch_encodings):
    logits = []
    for i in range(4):
        input_dict = {
            k: torch.cat([ex[i][k] for ex in batch_encodings], dim=0).to(DEVICE)
            for k in batch_encodings[0][i].keys()
        }
        out = self.bert(**input_dict)
        cls_embed = out.last_hidden_state[:, 0, :]
        logit = self.classifier(cls_embed)
        logits.append(logit)
    logits = torch.cat(logits, dim=1)
    return logits
```

---

### Results Table

| Metric                  | Value                        |
|-------------------------|------------------------------|
| **Validation Accuracy** | 64.4%                        |
| **Test Accuracy**       | 63.6%                        |
| **Training Time**       | 8691.42 seconds (~3 epochs)  |
| **Validation Time**     | 54.76 seconds                |
| **Test Inference Time** | 62.99 seconds                |
| **# Parameters Tuned**  | ~110M                        |

---

### Limitations and Future Improvements

- **Hardware Bottleneck:** All training was conducted on CPU, limiting training time and scalability. Using a GPU would significantly accelerate experiments and allow for more extensive tuning.
- **No hyperparameter search:** Learning rate, max sequence length, and batch size were chosen empirically and not tuned.
- **Long Input Truncation:** OpenBookQA sometimes has long `fact + stem + choice` combinations. With a `max_length` of 128, information may have been lost.
- **Simple classifier head:** Only a linear layer was added after the `[CLS]` token. Using more complex architectures could potentially improve performance.
