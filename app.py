
import pickle
import pandas
from flask import Flask, render_template, request, jsonify
import numpy as np
app = Flask(__name__, static_url_path='',
            static_folder='Static')
model = pickle.load(open('model.pkl', 'rb'))


@app.route('/')
def hello_world():
    return render_template('main.html')


@app.route('/predict', methods=['GET'])
def predict():

    age = request.args.get('age')
    sex = request.args.get('sex')
    cp = request.args.get('cp')
    trestbps = request.args.get('trestbps')
    chol = request.args.get('chol')
    fbs = request.args.get('fbs')
    resting = request.args.get('resting')
    thali = request.args.get('thali')
    exang = request.args.get('exang')
    oldpeak = request.args.get('oldpeak')
    slope = request.args.get('slope')
    ca = request.args.get('ca')
    thal = request.args.get('thal')

    # return render_template('main.html', pred='it is ')

    #    new_data = pd.read_csv(r'C:\Users\farah\OneDrive\Desktop\new_data.csv')

    #    X_new = new_data.iloc[:, :-1].values
    #    sc = StandardScaler()

    # Handling missing data for the new dataset
    # Calculate the number of missing features
    #    num_missing_features = X.shape[1] - X_new.shape[1]
    #    if num_missing_features > 0:
    #        imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    #        imputer = imputer.fit(X[:, -num_missing_features:])
    #        X_new[:, -num_missing_features:] = imputer.transform(
    #           X_new[:, -num_missing_features:])

    #X_new[:, 11:13] = imputer.transform(X_new[:, 11:13])
    # print("d")
    #    if X_new.shape[1] != X_train.shape[1]:
    #        num_extra_features = X_train.shape[1] - X_new.shape[1]
    #        X_new = np.append(X_new, np.zeros(
    #            (X_new.shape[0], num_extra_features)), axis=1)
    # Feature Scaling for the new dataset
    #    X_new = sc.transform(X_new)

    # Make predictions on the new dataset
    #    y_pred_new = classifier.predict(X_new)

    # Print the predictions
    # print(y_pred_new)
    #    return render_template('main.html', prediction_test"IT IS $ {}".format(y_pred_new))
    int_features = [int(age), int(sex), int(cp), int(trestbps), int(chol), int(fbs), int(
        resting), int(thali), int(exang), int(oldpeak), int(slope), int(ca), int(thal)]
    df = pandas.read_csv('cleve.csv')
    #int_features = [int(x) for x in request.form.values()]
    final = [np.array(int_features)]
    prediction = model.predict(final)

    decimal_part, integer_part = np.modf(prediction)

    integer_part = np.zeros_like(integer_part)

    decimal_part = np.round(decimal_part, 2)

    output = integer_part + decimal_part

    #output = '{0:.{1}f}'.format(prediction[0], 2)

    # if output < 0:
    #     output = output - output
    # else:
    #     output = (output +1)-output

    return render_template('result.html', pred='{}'.format(output))


if __name__ == "__main__":
    app.run()
