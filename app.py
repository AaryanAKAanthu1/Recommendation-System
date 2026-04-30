from flask import Flask, render_template, request
import recommender

app = Flask(__name__)
recommender = recommender.Recommender()

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/recommend", methods=["POST"])
def recommend():
    watched = {}
    watched['title'] = request.form["title"]
    watched['type'] = request.form["type"]
    watched['listed_in'] = request.form["genre"]
    watched['release_year'] = request.form["release_year"]
    watched['description'] = request.form["description"]
    watched['director'] = request.form["director"]
    watched['cast'] = request.form["cast"]
    watched['rating'] = request.form["rating"]
    watched['duration'] = request.form["duration"]

    recommendations = recommender.recommend([watched])
    titles = []
    limit = len(recommendations) if len(recommendations) < 10 else 10

    for i in range(0, limit):
        titles.append(recommendations[i]['title'])
    return render_template('index.html', recommendations=titles)


if __name__ == '__main__':
    app.run()