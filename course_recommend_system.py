import uvicorn
import asyncio
import threading
import nest_asyncio
import socket
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import os

# For Jupyter compatibility
nest_asyncio.apply()

# === FastAPI setup ===
app = FastAPI()
templates = Jinja2Templates(directory="templates")

# === Create template folder & HTML if doesn't exist ===
if not os.path.exists("templates"):
    os.makedirs("templates")

with open("templates/index.html", "w", encoding="utf-8") as f:
    f.write("""
<!DOCTYPE html>
<html>
<head>
    <title>Udemy Course Recommender</title>
</head>
<body>
    <h1>🎯 Udemy Course Recommender</h1>
    <form action="/search_course" method="get">
        <label for="query">🔍 Search for a course:</label>
        <input type="text" id="query" name="query" required>
        <button type="submit">Search</button>
    </form>
    {% if courses %}
        <h2>Search Results:</h2>
        <ul>
        {% for course in courses %}
            <li>
                <strong>{{ course['course_title'] }}</strong>
                <form action="/recommend" method="get" style="display:inline;">
                    <input type="hidden" name="course_index" value="{{ course['index'] }}">
                    <button type="submit">Recommend Similar</button>
                </form>
            </li>
        {% endfor %}
        </ul>
    {% endif %}
    {% if recommendations %}
        <h2>Recommended Courses:</h2>
        <ul>
        {% for course in recommendations %}
            <li><strong>{{ course['course_title'] }}</strong></li>
        {% endfor %}
        </ul>
    {% endif %}
</body>
</html>
    """)

# === Recommender Class ===
class UdemyCourseRecommender:
    def __init__(self, n_clusters=5, random_state=42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        self.course_data = None

    def load_data(self, file_path):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        if file_path.endswith('.csv'):
            self.course_data = pd.read_csv(file_path)
        elif file_path.endswith('.xlsx') or file_path.endswith('.xls'):
            self.course_data = pd.read_excel(file_path)
        else:
            raise ValueError("Unsupported file format")

        self.course_data.reset_index(inplace=True)
        return "Data loaded."

    def preprocess_and_cluster(self):
        numerical_data = self.course_data.select_dtypes(include=[np.number])
        scaled_data = self.scaler.fit_transform(numerical_data)
        self.kmeans.fit(scaled_data)
        self.course_data["cluster"] = self.kmeans.labels_

    def find_course_by_title(self, query):
        if self.course_data is None:
            return []
        matches = self.course_data[self.course_data['course_title'].str.contains(query, case=False, na=False)]
        return matches.to_dict(orient="records")

    def recommend_similar_courses(self, course_index, n=5):
        if self.course_data is None:
            return []
        cluster = self.course_data.loc[course_index, "cluster"]
        same_cluster = self.course_data[self.course_data["cluster"] == cluster]
        same_cluster = same_cluster[same_cluster["index"] != course_index]
        return same_cluster.head(n).to_dict(orient="records")

# === Instantiate and Load your Data ===
recommender = UdemyCourseRecommender(n_clusters=4)

# ✅ UPDATE this line with your actual file name
data_path = r"C:\Users\krish\Desktop\ML C2B2_2\udemy_course_data.csv"  # Correct file path here
recommender.load_data(data_path)
recommender.preprocess_and_cluster()

# === Routes ===
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/search_course", response_class=HTMLResponse)
def search_course(request: Request, query: str):
    courses = recommender.find_course_by_title(query)
    return templates.TemplateResponse("index.html", {
        "request": request,
        "courses": courses
    })

@app.get("/recommend", response_class=HTMLResponse)
def recommend(request: Request, course_index: int):
    recommendations = recommender.recommend_similar_courses(course_index)
    return templates.TemplateResponse("index.html", {
        "request": request,
        "recommendations": recommendations
    })

# === Auto Start Server on Random Free Port ===
def find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]

def start_server():
    port = find_free_port()
    print(f"🚀 Server running at: http://127.0.0.1:{port}")
    config = uvicorn.Config(app=app, host="127.0.0.1", port=port, log_level="info")
    server = uvicorn.Server(config)
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(server.serve())

server_thread = threading.Thread(target=start_server, daemon=True)
server_thread.start()
