from . import auth
#注册蓝图路由。
@auth.route('/')
def admin_index():
    return 'admin_index'