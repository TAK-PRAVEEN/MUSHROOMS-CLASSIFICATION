# 🍄 Mushroom Classification Web App

This project is a Machine Learning web application designed to predict whether a mushroom is **edible or poisonous** based on user inputs. It utilizes a custom pipeline and is built using Flask for the web interface.

---

## 📸 Preview
<img src="https://private-user-images.githubusercontent.com/132296372/455808743-f2aba241-c41e-48c0-81cd-47ef92d65151.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NTAxMzQwNDMsIm5iZiI6MTc1MDEzMzc0MywicGF0aCI6Ii8xMzIyOTYzNzIvNDU1ODA4NzQzLWYyYWJhMjQxLWM0MWUtNDhjMC04MWNkLTQ3ZWY5MmQ2NTE1MS5wbmc_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBVkNPRFlMU0E1M1BRSzRaQSUyRjIwMjUwNjE3JTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDI1MDYxN1QwNDE1NDNaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT04ODMyODE0OTFmY2M0ZDU5OGRiYzIyNDU2ZTQyOGNhMWRjMzViZTY5MmNmYjhkMWIxN2ZmNjM1MGVjN2U2YmNmJlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCJ9.nue4ePrLmSTQuOG5026zm0ATPZLDRSDe5n1yl_XlpTI" width="500"> <img src="https://private-user-images.githubusercontent.com/132296372/455808806-2190cf32-33f7-4d7c-8658-356f2266ed11.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NTAxMzQwNDMsIm5iZiI6MTc1MDEzMzc0MywicGF0aCI6Ii8xMzIyOTYzNzIvNDU1ODA4ODA2LTIxOTBjZjMyLTMzZjctNGQ3Yy04NjU4LTM1NmYyMjY2ZWQxMS5wbmc_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBVkNPRFlMU0E1M1BRSzRaQSUyRjIwMjUwNjE3JTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDI1MDYxN1QwNDE1NDNaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT05ZjdmZDQ1ZmE0OWFkNDllNDYzNWNhYWZjMWUyM2Y3NGQyMjY0NWU5MDJjOTIzZTFiZGIyM2VkZTQwMmI0MWE0JlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCJ9.7K-gbvpEdKkntAYgYgzCCEBOqcJFmzJwWR8J9FwlMCE" width="500">

---

## 🚀 Features

- User-friendly web interface for mushroom classification
- Custom ML pipeline using scikit-learn
- Deployable on platforms like Render or Heroku
- Lightweight and easy to extend

---

## 🧠 Model Details

- Input: Features like gill color, odor, and spore print color
- Output: Prediction of "Edible", "Poisonous", or "Not Match"
- Classification using a trained pipeline (`PredictPipeline`)

---

## 📁 Project Structure

```

MUSHROOMS-CLASSIFICATION/
│
├── app.py                  # Flask application
├── templates/              # HTML templates
├── src/                    # Source code for ML pipeline
├── notebook/               # Jupyter Notebooks for EDA & modeling
├── requirements.txt        # Python dependencies
├── setup.py                # Package configuration
└── artifacts/              # Trained models and outputs

```

---

## 🌐 Links

| Type            | URL (Placeholder)                           |
|-----------------|---------------------------------------------|
| 🔗 Live Demo    | [Render Link](https://your-render-link.com) |
| 📄 Documentation| [Docs](https://your-doc-link.com)           |

*Replace with actual URLs if available.*

---

## 📦 Setup Instructions

1. **Clone the repository**

```bash
git clone https://github.com/TAK-PRAVEEN/MUSHROOMS-CLASSIFICATION.git
cd MUSHROOMS-CLASSIFICATION
````

2. **Install dependencies**

```bash
pip install -r requirements.txt
```

3. **Run the app**

```bash
python app.py
```

---

## 🧑‍💻 Author

**Praveen Tak**
[GitHub Profile](https://github.com/TAK-PRAVEEN)

---

## 📜 License

This project is licensed under the MIT License.



