from flask import Blueprint, request, render_template, redirect, url_for, session, flash
from .extentions import mongo
import bcrypt

auth = Blueprint("auth", __name__)


@auth.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "GET":
        return render_template("flask_login.html", title="Login")

    users = mongo.db.users
    login_user = users.find_one({"email": request.form["email"]})

    if login_user:
        if bcrypt.checkpw(
            request.form["password"].encode("utf-8"), login_user["password"]
        ):
            session["id"] = str(login_user["_id"])  # add user to session
            return redirect(url_for("main.index"))
    flash("Invalid email or password. Please try again.", "danger")
    return redirect(url_for("auth.login"))


@auth.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "GET":
        return render_template("flask_login.html", register=True, title="Sign Up")

    users = mongo.db.users
    check_user = users.find_one({"email": request.form["email"]})
    if not check_user:
        hashed_password = bcrypt.hashpw(
            request.form["password"].encode("utf-8"), bcrypt.gensalt()
        )
        user = users.insert_one(
            {
                "name": request.form["name"],
                "password": hashed_password,
                "email": request.form["email"],
            }
        )
        session["id"] = str(user.inserted_id)  # add user to session
        flash("Account created successfully. Please proceed to login", "success")
        return redirect(url_for("auth.login"))
    else:
        flash("An account already exists with this email.", "danger")
        return redirect(url_for("auth.login"))


@auth.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("auth.login"))
