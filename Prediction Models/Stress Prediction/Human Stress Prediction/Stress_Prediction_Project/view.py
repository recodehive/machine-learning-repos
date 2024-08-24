from django.contrib.auth.forms import AuthenticationForm
from django.contrib.auth import authenticate, login
from django.http import HttpResponse
import pickle
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import string
from nltk.corpus import stopwords
from django.shortcuts import render, redirect
from django.http import HttpResponse
from django.contrib.auth.models import User
from django.contrib import messages
from django.core.mail import EmailMessage, send_mail
from Stress_Prediction_Project import settings
from django.contrib.sites.shortcuts import get_current_site
from django.template.loader import render_to_string
from django.utils.http import urlsafe_base64_decode, urlsafe_base64_encode
from django.utils.encoding import force_bytes, force_str
from django.contrib.auth import authenticate, login, logout
from .tokens import generate_token


# Load the model and transformers
with open('Stress_Prediction_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('CountVectorizer.pkl', 'rb') as cv_file:
    mess_transformer = pickle.load(cv_file)

with open('TfidfTransformer.pkl', 'rb') as tfidf_file:
    tfidf_transformer = pickle.load(tfidf_file)


def text_process(mess):
    nopunc = [char for char in mess if char not in string.punctuation]
    nopunc = ''.join(nopunc)

    return ' '.join([word for word in nopunc.split() if word.lower() not in stopwords.words('english')])


def home(request):
    return render(request, 'home.html')


def predict(request):
    if request.method == 'GET':
        return render(request, 'index.html')

    elif request.method == 'POST':
        if request.POST.get('predict_btn'):
            text = request.POST.get('text')
            processed_text = text_process(text)
            bow = mess_transformer.transform([processed_text])
            tfidf = tfidf_transformer.transform(bow)
            prediction = model.predict(tfidf)

            return render(request, 'index.html', {'prediction': f'The prediction is: {prediction[0]}'})

    return render(request, 'index.html')


def signup(request):
    if request.method == "GET":
        return render(request, "signup.html")

    if request.method == "POST":
        if request.POST.get('sign_up_submit'):

            username = request.POST['username']
            fname = request.POST['fname']
            lname = request.POST['lname']
            email = request.POST['email']
            pass1 = request.POST['pass1']
            pass2 = request.POST['pass2']

            if User.objects.filter(username=username):
                messages.error(request, "Username already exist! Please try some other username.")
                return redirect('signup')

            if User.objects.filter(email=email).exists():
                messages.error(request, "Email Already Registered!!")
                return redirect('signup')

            if len(username) > 20:
                messages.error(request, "Username must be under 20 charcters!!")
                return redirect('signup')

            if pass1 != pass2:
                messages.error(request, "Passwords didn't matched!!")
                return redirect('sigup')

            if not username.isalnum():
                messages.error(request, "Username must be Alpha-Numeric!!")
                return redirect('signup')

            myuser = User.objects.create_user(username, email, pass1)
            myuser.first_name = fname
            myuser.last_name = lname
            #myuser.is_active = False
            myuser.save()
            messages.success(request,
                             "Your Account has been created succesfully!! Please check your email to confirm your email address in order to activate your account.")

            return redirect('signin')

    return render(request, "signup.html")


def activate(request, uidb64, token):
    try:
        uid = force_str(urlsafe_base64_decode(uidb64))
        myuser = User.objects.get(pk=uid)
    except (TypeError, ValueError, OverflowError):
        myuser = None

    if myuser is not None and generate_token.check_token(myuser, token):
        myuser.is_active = True
        #user.profile.signup_confirmation = True
        myuser.save()
        login(request, myuser)
        messages.success(request, "Your Account has been activated!!")
        return redirect('signin')
    else:
        return render(request, 'activation_failed.html')


def signin(request):

    if request.method == 'GET':
        return render(request, "signin.html")

    elif request.method == 'POST':
        if request.POST.get('signin_btn'):

            username = request.POST['username']
            pass1 = request.POST['pass1']

            user = authenticate(username=username, password=pass1)

            if user is not None:
                login(request, user)
                messages.success(request, "Logged In Sucessfully!!")
                return redirect('predict')
            else:
                messages.error(request, "Bad Credentials!!")
                return render(request, "signin.html")

        return redirect('signin')

    return render(request, "signin.html")


def signout(request):
    logout(request)
    messages.success(request, "Logged Out Sucessfully!!!")
    return redirect('home')

