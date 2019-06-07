from flask import Blueprint

#Blueprint必须指定两个参数，auth表示蓝图的名称，__name__表示蓝图所在模块
auth = Blueprint('auth', __name__)
from example.chapter_4.auth import views
