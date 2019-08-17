from flask import Flask, flash, redirect, render_template, request, session, abort
import json
app = Flask(__name__)
# app.debug = True

@app.route('/')
def hello_world():
   return 'Hello World'
@app.route('/home')
def index():
   return render_template('home_page.html')
if __name__ == '__main__':
    app.run()