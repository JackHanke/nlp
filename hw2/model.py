import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoTokenizer, BertModel, GPT2LMHeadModel, GPT2Tokenizer

#
def test_model(model, test):
    model.eval()

    num_right = 0
    num_total = len(test)

    for question in test:
        pred = model.inference(questions=questions)
        if pred == answer: num_right += 1

    return num_right/num_total

# 
def train_model(model, train, epochs, optimizer, loss_fn):
    model.train()
    batch_size=5
    num_batches = len(train)//batch_size
    for epoch in range(epochs):
        # 
        prog_bar = tqdm(range(num_batches), total=num_batches)
        for batch_num in prog_bar:
            optimizer.zero_grad()
            # get data
            questions = train[(batch_size*batch_num): (batch_size*(batch_num+1))]
            # predictions
            pred = model.forward(questions=questions)
            # format answer tensor
            answer_tensor = torch.Tensor([q[1] for q in questions]).long().squeeze(1)

            train_loss_val = loss_fn(pred, answer_tensor)

            train_loss_val.backward()

            prog_bar.set_description(f'Epoch {epoch}, Batch {batch_num} Train loss: {train_loss_val:.6f}')

            optimizer.step()

        acc = test_model(model=model, test=valid)
        print(f'Epoch {epoch}, Batch {batch_num} Validation accuracy: {acc*100:.4f}')


# 
class QAModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.name='QAfinetuned_BERT'
        self.save_path = f'saves/{self.name}.pth'
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.bert_model = BertModel.from_pretrained("bert-base-uncased")
        self.linear = torch.rand(768, 1)

    def forward(self, questions: list[str]):
        pred_tensor = []
        for question in questions:
            # tokenize question
            tokens = self.tokenizer(question[0], return_tensors="pt", padding=True, truncation=True)
            # make batch predictions tensor
            output = self.bert_model(**tokens).last_hidden_state[:, 0].unsqueeze(0) # get embedding of [CLS] token
            pred_tensor.append(output)

        pred_tensor = torch.cat(pred_tensor)

        x = (pred_tensor @ self.linear).squeeze(2)
        return x

    @torch.no_grad()
    def inference(self, questions: list[str]):
        x = self.forward(questions=questions)
        val = torch.argmax(x)
        return val

    def save(self):
        torch.save(self.state_dict(), self.save_path)
        print(f'nsaved model as {self.save_path}')
