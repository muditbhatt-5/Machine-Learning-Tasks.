import os
import pandas as pd
from flask import Flask, request, render_template, redirect, url_for
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import joblib

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"
app.config["MODEL_FOLDER"] = "models"
app.config["ALLOWED_EXTENSIONS"] = {"csv"}

# Ensure directories exist
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
os.makedirs(app.config["MODEL_FOLDER"], exist_ok=True)

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in app.config["ALLOWED_EXTENSIONS"]

def train_models_and_generate_files(data_file):
    # Load dataset
    data = pd.read_csv(data_file)

    # Assuming the last column is the target variable
    X = data.iloc[:, :-1]  # Features
    y = data.iloc[:, -1]   # Target

    # Split dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define models
    models = {
        "Logistic Regression": LogisticRegression(max_iter=200),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(),
        "Gradient Boosting": GradientBoostingClassifier(),
        "SVM": SVC(probability=True),
        "K-Nearest Neighbors": KNeighborsClassifier(),
        "Naive Bayes": GaussianNB()
    }

    results = {}

    for name, model in models.items():
        # Train the model
        model.fit(X_train, y_train)

        # Save the model to a .pkl file
        model_filename = f"{name.replace(' ', '_')}.pkl"
        model_path = os.path.join(app.config["MODEL_FOLDER"], model_filename)
        joblib.dump(model, model_path)

        # Evaluate model performance
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        results[name] = accuracy

    return results

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Check if the file part is present in the request
        if "file" not in request.files:
            return redirect(request.url)

        file = request.files["file"]

        # Check if a file is selected and is allowed
        if file and allowed_file(file.filename):
            file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(file_path)

            # Train models and generate results
            results = train_models_and_generate_files(file_path)

            # Identify the best model based on accuracy
            best_model = max(results, key=results.get)
            best_accuracy = results[best_model]

            return render_template(
                "index.html", results=results, best_model=best_model, best_accuracy=best_accuracy
            )

    return render_template("index.html", results=None)

if __name__ == "__main__":
    app.run(debug=True)
