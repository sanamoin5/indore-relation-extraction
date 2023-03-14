from transformers import RobertaTokenizer, RobertaForTokenClassification
import torch

tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
model = RobertaForTokenClassification.from_pretrained('roberta-base')

inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
print(inputs)

labels = torch.tensor([1] * inputs["input_ids"].size(1)).unsqueeze(0)  # Batch size 1
print(labels)
print([1] * inputs["input_ids"].size(1))

outputs = model(**inputs, labels=labels)
print(outputs)
loss = outputs.loss
print(loss)
logits = outputs.logits
print(logits)
predicted_class_ids = torch.argmax(outputs.logits, dim=-1)
print("This is is Class ID's")
print(predicted_class_ids)
predicted_label = model.config.id2label[predicted_class_ids]
print("This is is Label")
print(predicted_label)



def tokenize_and_encode_data(self):
    encoded_dict = self.tokenizer.encode_plus(

        # TODO: Create a Batch of Sentences to pass it to the model.
        "<s> Hello <e1>is</e1> Brother you <e2>are</e2> great. "
        "My Name is <e1>Navneet</e1> friend of <e2>Sushil</e2> </s>",
        add_special_tokens=False,

        # TODO: Need to discuss about the maximum tokenization length
        max_length=100,
        padding=True,
        truncation=True,
        return_attention_mask=True,

        # return Pytorch Tensors.
        return_tensors='pt'
    )

    # print("This is encoded dictionary")
    # print(encoded_dict)
    # inputs = self.tokenizer("<s> Hello <e1>Navneet</e1>. Brother you and <e2>Sushil</e2> great </s>")
    # print(self.tokenizer.decode(inputs["input_ids"]))
    # # print(self.tokenizer.tokenize("Hello Brothers!"))
    #
    # # labels = torch.tensor([1] * encoded_dict["input_ids"].size(1)).unsqueeze(0)  # Batch size 1
    # labels = torch.tensor(['brother']).unsqueeze(0)
    # # # print("This is Labels")
    # # # print(labels)
    # #
    # outputs = self.model(**encoded_dict, labels=labels)
    # # # print("This is Outputs")
    # print(outputs)
    # # # loss = outputs.loss
    # # # print("This is Loss")
    # # # print(loss)
    # # # logits = outputs.logits
    # # # print("This is Logits")
    # # # print(logits)
    # #
    # # predicted_class_ids = torch.argmax(outputs.logits, dim=-1)
    # # print("This is is Class ID's")
    # # print(predicted_class_ids)
    # # predicted_label = self.model.config.id2label[predicted_class_ids]
    # # print("This is is Label")
    # # print(predicted_label)