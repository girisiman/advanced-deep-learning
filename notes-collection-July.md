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
- Key value retireivall networks for Task oriented dialogues - {paper}
- checkout cactus needle dataset add (mix) in the translation dataset or dailogue 
- ontologies based knwledge graph { json based tool call} for dialogue and tranlation models
- 
# 6 August Notes

## Topics [ Contd from Bahdanau Attentions]
- Standard RNN
  - $ f_h(H, I) -> H' $
  - $ f_o (H', I) -> O or f_o(H, I) -> O or f_o(H') -> O$
- next - Recurrent Transformer Architecture { Idea Replace RNN/LSTM with Transformer but in LSTM like recurrent architecture}
- { Are Transformer Inspired by Reidual Network?}
- {kaming He - ResNet talk key note}
- KV cache
- Parameter Cost
  - MLP is most expensive (simplest but most inefficient)
  - Assignment - In general translation model use cactus needle architecture { RNN - Individual block is Cactus Needle, also try resnet with individual block could be cactus needls}
- Neural machine Translation Models
  - Enc{F_ENC(I) -> H} - Dec Architecture 
  - Dec only
  - Initial Hidden state (for recurrent) could be some learnable embedded table instead of Random initializations {for coding considerations}
  - Trainig Regimes (considerations) Setup:
    - Pre training 
      - NLP -> Predict Next token
    - fine training (Supervised fine tuning)
      - Task specific (translation)
      - prompt based (ASR)
    - 1 pretraining + sft
    - 2 sft (promped based) use next token as prompt this is how we can combine (both to sft cause sft would not work on ASR alone)
  - Challenges of Tokenization
    - Currently we only have BPE {similar to Hoffman coding}
      - BPE Idea - most frequent subsequence into one digit - {})
    - (Dynamic Embedding Table - just a possibility - go to character based model ?)
- Build a promtable translation model mixing with cactus needle for language
- Some example prompts:
  - predict the next word
  - Translate in language x
  - Translate into mixed language x & y (define amount of mixing)
  - can be update for diverse set of task not limited to translation, also can be done fill in the blanks, 
  - for e.g. Eng - Nepali translation we can create 10 different task (like fill in the blanks), generate the dataset then do pre or prompt based training
  - more precisely:
    - create pre - training data, -> create different task
    - create sft data -> create different tasks
    - select data point, -> create random prompts
    - Train all togethers using Categorical cross entropy loss 
- some tool calling task { what is the time? generate some JSON}
- Experiment with architectures
- Experiment with pre training, fine tuning (sft), 
- Recurrent Transformer and Standard Resnet (experiment with different architectures)
- in resnet have fixed set of blocks lets say 5 they all have fixed set of parameters update happens at once
- next make folded or looped like rnn 
- In resnet - we have a loop but loop also comes back but have the gate (sigmoid) that defines whether it is output or needs further processing go down the loops fixed the loops number lets say 10 { looks like mixed of above two} 
  - figure out how to train? tricky 
  - 
  - gate will only give output 1 (i.e. if sigmoid output 1 we start backpropagation)
    - { Getting towards HRM} 

# 13 August Notes
## ADL Topics Covered
- Flow matching & Diffusion
- distillation (large to small , vs self distillation,)
- Self read:
  - GAN
  - Activation Function (recent development) - GLU, SWIGLU, sigoid activation units
  - Normalization (recent devlopment) - Layernorm,Batch Norm, RMS Norm
  - MoE, General Adapters
- Energy Based Models  {prob vs Energy; Lack of Normalizations, recent advances, model collapse}
- Basic VLA
- This is the Plan
  - How to get rid of MLP Layers in Transformer
  - How to add memory vis {KV} Mixture of Adapters

# Flow Matching
  - Traing ConvEncoder, extract latent pass it to Flowmatching, 
  - Inference time sample x_0 (gaussian) padd it to Decoder to generate new image
  - {one of the way to implement flow matching for image generation}