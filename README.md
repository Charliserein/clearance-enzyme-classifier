# clearance-enzyme-classifier


model

RNN Bilstm XGBoost

The environment is specified in the requirements.txt file.

Output The final classification results are stored in 'final_result.csv'.
![e63e60142881c24aaa02e7b095aa77f](https://github.com/user-attachments/assets/473f2b76-cb21-415c-b7d7-1bf915fcea28)
![image](https://github.com/user-attachments/assets/e65a2dd2-8b02-4551-9317-227bf277393c)
![image](https://github.com/user-attachments/assets/80ef5596-4843-4c6b-acf5-0cb2b5875b06)
![image](https://github.com/user-attachments/assets/58778015-3262-48df-90fb-d9f6950f581c)
![image](https://github.com/user-attachments/assets/fd96f4c9-0708-41b9-bbf4-4de621adf3ba)

we present the accuracy and recall of the three algorithms in classifying different categories (we randomly selected 20 categories). RNN-attention: This model shows high stability in both accuracy and recall, making it suitable for tasks that strongly rely on temporal or sequential information. Its fluctuations are relatively small; although there was a significant drop during the 15th iteration, it quickly recovered and performed excellently overall. BiLSTM: While it performs outstandingly in accuracy, close to RNN-attention, the recall shows larger fluctuations, particularly with notable drops during the 10th and 15th iterations, possibly due to instability in handling some difficult samples. Nevertheless, it remains a very robust model overall. XGBoost: This model performed slightly worse in early training; although its accuracy gradually caught up with the other models, it showed significantly greater volatility. The notable drop during the 15th iteration suggests that this model may be sensitive to certain patterns or features in the training dataset. For tasks requiring high stability, XGBoost may not perform as well as the first two models.

![image](https://github.com/user-attachments/assets/d2b15252-a1f0-436d-84c8-b66b80ef9e3c)
![image](https://github.com/user-attachments/assets/d4151cf4-54ac-4171-b104-df7680257443)
