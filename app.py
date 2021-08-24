from flask import Flask, render_template, url_for, flash, redirect, request

from forms import RegistrationForm, LoginForm

from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from flask_login import LoginManager


app = Flask(__name__)
bootstrap = Bootstrap(app)
app.config['SECRET_KEY'] = 'fit3164'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///site.db'

bcrypt = Bcrypt(app)
db = SQLAlchemy(app)
login_manager = LoginManager(app)


from models import User

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))



@app.route("/home.html")
def home():
    return render_template('home.html', title = 'Home')

@app.route("/aboutus.html")
def about():
    return render_template('aboutus.html', title = 'About Us')

@app.route("/predictivemodel.html", methods=['GET', 'POST'])
def predictivemodel():
    form = RegistrationForm()
    if form.validate_on_submit():
        hashed_password = bcrypt.generate_password_hash(form.password.data).decode('utf-8')
        user = User(username=form.username.data, email=form.email.data, password=hashed_password)
        db.session.add(user)
        db.session.commit()
        flash('Your account has been created! You are now able to log in', 'success')
        return redirect(url_for('login'))
    return render_template('predictivemodel.html', title='Register', form=form)

@app.route("/login.html", methods=['GET', 'POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        return redirect(url_for('about.html'))

        
    return render_template('login.html', title='Login', form=form)
      
@app.route("/understandingpage.html")
def understandingpage():
    return render_template('understandingpage.html', title = 'Understanding Breast Cancer')



if __name__ == '__main__':
	app.run(debug=True)