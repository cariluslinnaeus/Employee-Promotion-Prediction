from flask import Flask,render_template,request,url_for
import employees_promotion_ml as em




app = Flask(__name__)
@app.route("/")
def hello():
    return render_template("employee.html")
@app.route("/prediction", methods = ["POST"])
def predictions():
    if request.method == "POST":
        department = int(request.form.get("department"))
        education = int(request.form.get("education"))
        gender = int(request.form.get('gender'))
        no_of_training= int(request.form["training"])
        age = int(request.form["age"])
        rating = int(request.form["rating"])
        experience = int(request.form["service"])
        kpis_met = int(request.form["KPI"])
        awards_won = int(request.form["Award"])
        score = int(request.form["score"])
        sum_metric = awards_won + rating + kpis_met
        total_score = no_of_training*score

        prediction = em.prediction(department,education,gender,no_of_training,age,rating,experience,kpis_met,awards_won,score,sum_metric,total_score)

        

    return render_template("alert.html",result = prediction)

if __name__ == "__main__":
    app.run(debug=True)
