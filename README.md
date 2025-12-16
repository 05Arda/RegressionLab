🧪 Ultimate ML Lab: Interactive Regression Simulator

Ultimate ML Lab is a Python-based interactive educational tool designed to visualize and experience fundamental Machine Learning concepts.

With this lab environment, you can experiment with abstract concepts such as Bias-Variance Tradeoff, Overfitting, Regularization, and Cross-Validation on live graphs.

(You can add a screenshot of your application here)

🚀 Features

The application consists of 3 main control panels:

1. Model Parameters (Blue Panel)

Degree: Adjust the complexity (polynomial degree) of the model.

Alpha (Regularization): Set the penalty coefficient for Ridge regression. Ideal for reducing high variance.

2. Data Generation (Green Panel)

Dataset Types: Choose from Sine Wave, Linear, Step Function, or Heteroscedastic (variable variance) data.

Noise and Size: Control the size of the dataset and the amount of random noise added to it.

Randomness: Try infinite variations with the "New Random Seed" button.

3. Analysis and Tools (Orange Panel)

Complexity vs Learning Curves: Visualize error analysis based on "Model Complexity" or "Data Size".

Cross-Validation: Toggle 5-fold cross-validation for a more reliable measurement of model performance.

Manual Outlier Addition: Add your own outliers by clicking on the graph and watch the model's reaction.

🛠️ Installation

Follow these steps to run the project on your local machine:

Clone the Repository:
```shell
git clone https://github.com/05Arda/RegressionLab.git
cd ultimate-ml-lab
```
Create a Virtual Environment (Recommended):
```shell
python -m venv venv
```

# For Windows:
```shell
venv\Scripts\activate
```

# For Mac/Linux:
```shell
source venv/bin/activate
```

Install Requirements:
```shell
pip install -r requirements.txt
```

▶️ Usage

After installation is complete, enter the following command in the terminal to start the application:
```shell
python main.py
```

The application window will open. You can start experimenting using the sliders and buttons on the graphs.

📂 Project Structure
```
regressionlab/
│
├── main.py # Main application code (Python & Matplotlib)
├── requirements.txt # Required Python libraries
├── documentation.html # In-app help documentation
└── README.md # Project description (This file)
```
🤝 Contributing

Feel free to send a "Pull Request" for bug fixes or new feature suggestions!

Fork the Project.

Create a new Branch (git checkout -b feature/NewFeature).

Commit your changes (git commit -m 'Added new feature').

Push to the Branch (git push origin feature/NewFeature).

Open a Pull Request.

📄 License

This project is licensed under the MIT License.

Developer: 05Arda
