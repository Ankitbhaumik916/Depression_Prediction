# MindScope 🧠 - Student Depression Prediction

MindScope is an AI-powered system built to help detect signs of depression among students using a logistic regression model trained on real-world survey data.\
It aims to promote mental health awareness by offering early risk detection, in a simple and private way.

---

## ✨ Features

- Predicts risk of depression based on lifestyle, study habits, and emotional health.
- User-friendly interaction directly through the terminal.
- Cool visualizations to show probability trends and risk factors.
- Clean and lightweight model - No heavy ML libraries needed.
- Normalizes user data automatically to match model expectations.
- Protects from overflows, missing data, and input mismatches.
- Fully offline - Your data stays private.

---

## ⚙️ Tech Stack

- **Language:** Python 3

- **Libraries:**

  - NumPy
  - Pandas
  - Matplotlib (for visualizations)
  - Pickle (for model saving/loading)

- **ML Model:** Logistic Regression (built from scratch 💻)

---

## 📦 Installation

1. **Clone this repository:**

   ```bash
   git clone https://github.com/your-username/MindScope.git
   cd MindScope
   ```

2. **Install dependencies:**
   (Make sure you have Python 3 installed)

   ```bash
   pip install numpy pandas matplotlib
   ```

3. **Run the project:**

   ```bash
   python mindscope.py
   ```

---

## 🚀 How It Works

- The system trains a logistic regression model on the given dataset (`student_depression_dataset.csv`).
- It saves the trained weights and bias into a `.pkl` file.
- The user is prompted to input their personal and academic details.
- The model predicts if the user is at risk for depression based on these inputs.
- A bar graph displays the influence of different features.
- The system then displays friendly suggestions based on the prediction outcome.

---

## 📋 Example

```bash
Do you want to predict your depression risk? (yes/no): yes

Gender: Male
Age: 20
City: Kolkata
Profession: Student
Academic Pressure: 4
Work Pressure: 3
CGPA: 8.2
Study Satisfaction: 3
Job Satisfaction: 4
Sleep Duration: 6
Dietary Habits: Balanced
Degree: B.Tech
Have you ever had suicidal thoughts ?: No
Work/Study Hours: 6
Financial Stress: 2
Family History of Mental Illness: No
```

✅ Prediction: No signs of depression detected. Stay positive! 🌟

⬆️ Visualization: A bar graph showing top factors influencing mental health.

---

## 📈 Future Improvements

- Add a GUI using **Tkinter** or **PyQt**.
- Improve prediction accuracy using more advanced models (like SVM, Random Forest).
- Build a web app version using **Flask** or **Django**.
- Encrypt user inputs for full privacy.
- Add animated visualizations and a dashboard for easier understanding.

---

## 💬 A Note from Ankit

> *"Mental health matters. MindScope is a small step toward making it easier for students to get aware about their emotional well-being.*\
> *Stay positive, stay strong! ❤️"*

---

## 🔥 Made with love and Python by [Ankit Bhaumik](#) ❤️

