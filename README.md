# Financial Market Prediction

> A repository for implementing time-series forecasting models for stock market data.

---

We are experimenting with the following models:

1.  **ARIMAX (Autoregressive Integrated Moving Average with Exogenous Variables)**
2.  **LSTM (Long Short-Term Memory)**
3.  **TFT (Temporal Fusion Transformer)**
4.  **Reinforcement Learning (RL)**

---

## Our Team's Git Workflow Guide 

Hello team\! To make sure we all work together smoothly and keep our project organized, please follow this guide for every new task you work on. The goal is to keep our **`main`** branch clean, stable, and always working.

### The Workflow: A Step-by-Step Guide

#### Step 1: Start from an Up-to-Date `main` Branch

Before you start any new work, make sure you have the latest version of the project.

```bash
# Switch to the main branch
git checkout main

# Pull the latest changes from GitHub
git pull origin main
```

#### Step 2: Create Your New Branch

Create a new branch for your specific task. **Do not work directly on the `main` branch.**

Use our naming convention: **`type/short-description`**

  * **`type`**: Can be `feature`, `fix`, `docs`, `experiment`, etc.
  * **`short-description`**: 2-4 words describing the task, separated by hyphens.

<!-- end list -->

```bash
# Example: Create a branch to add the LSTM model
git checkout -b feature/implement-lstm-model
```

#### Step 3: Do Your Work & Commit Often 

Now you can work on your task (writing code, creating notebooks, etc.). Save your progress by making small, frequent commits with clear messages.

```bash
# Stage your changed files
git add .

# Commit them with a descriptive message
git commit -m "fAdd initial LSTM model architecture"
```

#### Step 4: Push Your Branch to GitHub 

When you're ready to share your work or open a pull request, push your branch to the remote repository.

```bash
# The '-u' flag is only needed the first time you push a new branch
git push -u origin feature/implement-lstm-model
```

#### Step 5: Open a Pull Request (PR) 

Go to the project's GitHub page. You will see a prompt to open a **Pull Request** for the branch you just pushed.

  * A PR is a formal request to merge your work into the **`main`** branch.
  * Give it a clear title and a short description of what you did.
  * Assign at least one teammate to review your code.

#### Step 6: Review and Merge 

  * **For the Author**: If your reviewer requests changes, make them on your branch and push them. The PR will update automatically.
  * **For the Reviewer**: Look through the code, ask questions, and provide feedback. If everything looks good, approve the PR.

Once the PR is approved, the author can merge it into **`main`**. Be sure to use the **"Squash and Merge"** option and **delete the branch** after merging.

That's the entire process\! Following these steps will help us stay organized and build our project effectively.








## Getting Started

To get a local copy up and running, follow these simple steps.

### Prerequisites

Make sure you have Python 3.8+ and pip installed.

### Installation

1.  Clone the repo
    ```sh
    git clone https://github.com/gourishankar27/financial-market-prediction.git
    ```
2.  Navigate to the project directory
    ```sh
    cd financial-market-prediction
    ```
3.  Install the required packages
    ```sh
    pip install -r requirements.txt
    ```

---