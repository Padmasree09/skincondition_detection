Skin Condition Detection and Product Recommendation System

A web application that detects skin conditions from uploaded images using a fine-tuned MobileNet CNN model and recommends suitable skincare products based on user skin type and detected concerns.

Features

    Upload a skin image for condition detection

    Predicts skin condition using MobileNet-based CNN

    Recommends skincare products matching skin concerns

    Flask-based lightweight web app

    Clean and simple UI


Tech Stack

    Frontend: HTML5, CSS3, Javascript (Flask templates)

    Backend: Python Flask

    Machine Learning: TensorFlow/Keras (MobileNet)- Deep learning CNN model

    Recommendation System: Collaborative based filtering system

    Dataset: CSV-based product dataset, Images(acne,comedone,clear) dataset


How to Download and Run the Application
1. Clone the Repository

   
       git clone https://github.com/Padmasree09/skincondition_detection.git
       cd skincondition_detection

   
3. Create a Virtual Environment (Recommended)


       python -m venv venv
       source venv/bin/activate        # Linux/Mac  (or)
       venv\Scripts\activate           # Windows

   
5. Install Required Packages


       pip install -r requirements.txt
   This will install packages like Flask, TensorFlow, NumPy, Pandas, Scikit-learn, etc.

   
7. Run the Flask App


       python app.py
   Flask server will start.
Visit http://localhost:5000 in your browser.


8. Upload Image and Get Prediction

    Upload your skin image.

    Select your skin type (e.g., Oily, Dry, Combination, Sensitive).

    Submit to get: Predicted skin condition, Recommended products

Requirements

    Python 3.8+

    Flask

    TensorFlow

    Scikit-learn

    Pandas

    NumPy

Future Improvements

    Deploy on cloud (AWS, Azure)

    Improve dataset size for better model accuracy

    Add user login/signup functionality


License

This project is licensed under the MIT License.

