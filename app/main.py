from flask import (
    Blueprint,
    redirect,
    render_template,
    session,
    url_for,
    jsonify,
    request,
    abort,
)
from .extentions import mongo
import bson

main = Blueprint("main", __name__)

@main.route("/")
def index():
    if "id" in session: # if user is logged in
        users = mongo.db.users
        user = users.find_one({"_id": bson.ObjectId(session["id"])})
        return render_template("index.html", name=user["name"])
    return redirect(url_for("auth.login"))

@main.route("/tracker")
def tracker():
    return render_template("tracker.html")

@main.route("/manager")
def manager():
    return render_template("manager.html")

@main.route("/get-data")
def get_data():
    records_db = mongo.db.records
    records = records_db.find({"user_id": session["id"]}).sort("dob", -1)
    results = []
    for record in records:
        record["_id"] = str(record["_id"])
        results.append(record)
    return jsonify(results), 200

@main.route("/add-data", methods=["POST"])
def add_data():
    data = request.get_json()
    records = mongo.db.records
    try:
        records.insert_one(
            {
                "user_id": session["id"],
                "name": data["name"],
                "quantity": data["quantity"],
                "purchase_price": data["purchase_price"],
                "dob": data["dob"],
            }
        )
        return jsonify({"success": True})
    except Exception as e:
        print(e)
        return jsonify({"success": False})

@main.route("/delete-data", methods=["DELETE"])
def delete_data():
    id_ = request.args.get("id")
    if not id_:
        abort(400)
    records = mongo.db.records
    try:
        records.delete_one({"_id": bson.ObjectId(id_)})
        return jsonify({"success": True})
    except Exception as e:
        print(e)
        return jsonify({"success": False})