# VerveBridge-ML-Internship
**TASK 1: Book Recommendation System**
Overview:
This project focuses on developing a book recommendation system that utilizes machine learning algorithms to suggest books based on user preferences. Built using Python, it leverages libraries like Pandas, Matplotlib, Seaborn, and Scikit-learn for data analysis and visualization. The system is deployed with Flask, a lightweight web framework, and utilizes Flask-CORS for cross-origin resource sharing.

Workflow:
Data Exploration and Analysis (EDA)
Data Preprocessing
Model Building
Database Integration
API and Web Interface
Logging and Error Handling
Environment:
To run the project, set up a Conda environment using the provided environment.yml file.

Channels: defaults
Dependencies: Python, Pandas, Matplotlib, Seaborn, Scikit-learn, Flask, Flask-CORS
Prefix: C:\Users\PythonCoding\Anaconda3\envs\Book-Rec-env
How to Run:
Create a new Conda environment: conda env create -f environment.yml
Activate the environment: conda activate Book-Rec-env
Navigate to the project directory: cd path/to/project/directory
Run the Flask application: python app.py
Access the book recommendation system via a web browser at http://localhost:5000.
Features:
Book Search and Filtering
Personalized Recommendations based on user preferences
Visualization of user reading habits and book ratings
Dataset:
You can download the dataset from here.

Contributing:
To contribute, fork the repository and submit a pull request. You can also report issues or suggest new features by creating an issue in the repository.

**TASK 2: ML Price Negotiator Chatbot System**
Overview:
The Price Negotiator Ecommerce Chatbot is designed to simulate real-world price negotiations within an e-commerce environment. By utilizing natural language processing (NLP), the chatbot engages with users, understands their queries and bargaining intents, and provides a dynamic pricing experience. The system integrates several services, including product catalog management, order management, and price negotiation logic, to enhance online shopping.

Key Features:
Natural Language Processing: Understands user inputs for price negotiations.
Dynamic Pricing: Adjusts product prices based on negotiation strategies.
Product Catalog Management: Maintains and queries available products.
Order Management: Handles order placements and updates.
Modular Design: Organized into independent services for scalability and ease of maintenance.

**TASK 3: Campus Placement Prediction**
Overview:
This project aims to predict whether a student will be placed in a campus recruitment drive based on various academic and extracurricular factors. By leveraging machine learning models, the system helps in understanding the factors that contribute to campus placements.

Features:
Multiple ML Models: Logistic Regression, Decision Tree, and Random Forest models are trained and evaluated.
Web Interface: User-friendly application where users can input student data and receive placement predictions.
Feature Engineering: Advanced techniques are implemented to enhance model performance.
Scalable and Modular Code: The project is structured modularly, making it easy to maintain and extend.
Dataset:
The dataset includes columns like:

sl_no: Serial Number
gender: Gender of the student (Male=0, Female=1)
ssc_p: Secondary Education percentage (10th Grade)
ssc_b: Board of Education (Central/State)
hsc_p: Higher Secondary Education percentage (12th Grade)
hsc_b: Board of Education (Central/State)
hsc_s: Specialization in Higher Secondary Education
degree_p: Degree Percentage
degree_t: Type of Undergrad Degree (Science/Commerce/Others)
workex: Work Experience (Yes/No)
etest_p: E-test Percentage
specialisation: Post Graduate Specialization (Marketing & HR/Finance)
mba_p: MBA Percentage
status: Placement Status (Placed/Not Placed)
salary: Salary offered to the student
Installation:
Clone the repository: git clone https://github.com/yourusername/campus-placement-prediction.git
Navigate to the directory: cd campus-placement-prediction
Create a virtual environment: python -m venv env
On Windows, use env\Scripts\activate; on other systems, use source env/bin/activate
Install the dependencies: pip install -r requirements.txt
Run the application: python app.py
Usage:
Web Interface: Open your browser and navigate to http://127.0.0.1:5000/. Enter the required student details and click "Predict" to see the placement prediction.
Model Evaluation: Use the evaluate.py script to evaluate the performance of the trained models.
