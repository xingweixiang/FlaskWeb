from flask import Flask, request, redirect, url_for, render_template
from example.chapter_4.auth import auth
app = Flask(__name__)


# @app.route('/')
# def hello_world():
#     return 'Hello World!'

@app.route('/hello/<name>')
def hello_name(name):
   return 'Hello %s!' % name

@app.route('/login',methods = ['POST', 'GET'])
def login():
   if request.method == 'POST':
      user = request.form['nm']
      return redirect(url_for('success',name = user))
   else:
      user = request.args.get('nm')
      return redirect(url_for('success',name = user))
@app.route('/')
def student():
   return render_template('student.html')
@app.route('/result',methods = ['POST', 'GET'])
def result():
   if request.method == 'POST':
      result = request.form
      return render_template("back_result.html",result = result)


#url_prefix是指在定义视图函数url前面加上/auth才能访问该视图函数
app.register_blueprint(auth,url_prefix='/auth')
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80)#host='0.0.0.0'，允许外网访问，端口为80
