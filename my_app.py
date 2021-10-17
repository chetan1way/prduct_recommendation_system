# Flask utils
from flask import Flask, request, render_template
from model import item_for_user


# Define a flask app
app = Flask(__name__)


@app.route('/')
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        Username = request.form['Username']
        items = item_for_user(Username)
        if items == 'User not found':
            return render_template('index.html', user=Username + ' ' + items)
        else:
            return render_template('index.html', user= 'Recommended Products for ' + Username ,product1 = items[0],product2= items[1],product3 = items[2],product4 = items[3],product5 = items[4] )
    else :
        return render_template('index.html')


if __name__ == '__main__':
    print('*** App Started ***')
    app.run(debug=True)