Face Recognition Project
========================

Run the full workflow from PowerShell:

    .\face_detection_project\run_project.cmd

The application asks for a name, captures 50 face images, retrains the name and
gender models, and opens the webcam recognition window.

The recognition window displays the recognized name and name accuracy. Gender
prediction is enabled only when `models/gender_model.h5` contains a trained
model built from labeled male and female face images.

To prepare automatic gender prediction, capture labeled samples for both
classes and train the gender model:

    ..\venv\Scripts\python.exe src\capture_gender_data.py
    ..\venv\Scripts\python.exe src\capture_gender_data.py
    ..\venv\Scripts\python.exe src\train_gender_model.py
