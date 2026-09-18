# Advance Deep Learning

This contains scribble of notes taken during ADL class.

## 🗓️ Lecture : [Sequence Model]
* **Date:** [30 - July]
* **Topics Covered:** [RNN, LSTM, ]

---

# 30 July

## 📝 Core Concepts & Theory

- Recurrent nets


## 📝 Task

- Implement very simple Vanilla RNN or Standard RNN without memory and demonstrate the problem of forgetting for longer sequence
- Idea is we will replace it later with 
- {concatenate : for hidden state - ct use w1ct-1+w2 and use differnet conctenatw for output for currentstate }
- {one concatenate then have parallel network one for hidden state and one for computing output state }
- For LSTM; f_c(Ct, h_(t-1)) f_delta(ct, h_(t-1)) such that C_t = c_t-1 + deltac_t, h_t - f_t-1) + deltah_t
- lets think cell state and hidden state are same;  
- Abstract understanding for LSTM and GRU , exactly what is happening in each gates
- LSTM netwirks in sequence labelling applications, {ref paper - Optimal Hyperpaprameters for Deep LSTM Networks for sequence labelling task}

---

## Encoder Decoder Architecture

- For language translation task 
- Nep - Eng translation task - {use same tokenizer, Experiments - Use pre- trained embedding, Gemma Tokenizer and also Embedding Table, Qwen Tokenizer or Embedding Table}
- write the corresponding equaiton for Seq to Seq model (make ickma style notes for LSTM and also for Seq to Seq Model)
- one enc output to one dec inp
- compressed enc output to all dec inp
- 2D projection of latent encoding of sample sentences (from sutsveker paper see plot)
- Experiment: add a prefix(a promt at beginging of encoder) 
- Eg: Translate into Nepali, John likes Mary 
- Eg: Translate into English, 
- {idea a promptable model that can perform Nepali to English and English to Nepali} with single model
- BLUE Score to measure the model performance {Bag of words}
  -  Union over Intersection - Unigram 
  -  Union over Intersection - Bigram 

- Sequence Architectures - {one to many vs. many - to - one vs many to many}

## Attention based Models

- Implement Bahdanau Attention 
- also make the notes like ickma style { image from my slides on the transformer can be used}
- Embedding and tokenizer are frozen {design some experiment}
- Key value retireivall netwoeks for Task oriented dialogues - {paper}
- checkout cactus needle dataset add in the translation dataset or dailogue 
- ontologies based knwledge graph { json based tool call} for dialogue and tranlation models

## 6 August Notes

- 