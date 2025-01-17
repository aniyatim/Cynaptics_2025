# Cynaptics_2025

## Overview

This repository showcases three machine learning developments for Cynaptics inductions.

---

## **Task 1**

## *Sub_Task 1: AI_vs_Real*

This task involves training a **Convolutional Neural Network (CNN)** to classify images as either **AI-generated** or **real**. The pipeline includes:
- **Data Preprocessing**: Preparing the dataset for training, including image resizing and normalization.
- **Model Building**: A sequential CNN model is constructed with layers designed for image classification.
- **Training**: The model is trained with **early stopping** to prevent overfitting and improve generalization.

## *Sub_Task 2: GAN*

This task implements a **Generative Adversarial Network (GAN)** for creating synthetic images. It includes:
- **Generator Network**: The generator creates synthetic images.
- **Discriminator Network**: The discriminator evaluates the images, distinguishing between real and fake images.
- **Adversarial Training**: The two networks are trained together in an adversarial setup on the **Fashion MNIST** dataset, iteratively improving the realism of the generated images.

---

## **Task 3: Finetuning LLM**

In this task, we fine-tune a **Large Language Model (LLM)** using **Low-Rank Adaptation (LoRA)** for dialogue generation. The process includes:
- **Pre-trained Falcon Model**: We start with the Falcon LLM and adapt it for specific use cases.
- **Fine-Tuning with LoRA**: Using Low-Rank Adaptation, we efficiently fine-tune the model to generate contextually appropriate responses.
- **Dialogue Input**: The fine-tuned model responds based on persona and dialogue input, ensuring dynamic, human-like conversations.

---
