from flask import Blueprint, render_template

bp = Blueprint('main', __name__, url_prefix='/')

@bp.route('/index')
def index():
    return render_template('index.html')

@bp.route('/')
def info():
    return render_template('info.html')